import json
import logging
import time
from uuid import UUID

import httpx
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import settings
from src.models.news import NewsArticle
from src.models.scoring import Direction, JevPromptContext, JevScore

logger = logging.getLogger(__name__)


class JevClient:
    def __init__(self):
        self.primary_client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=settings.jev_request_timeout,
        )

        # Failover: Vercel AI Gateway
        self.failover_client = None
        if settings.vercel_ai_gateway_url and settings.vercel_ai_gateway_token:
            self.failover_client = AsyncOpenAI(
                api_key=settings.vercel_ai_gateway_token,
                base_url=settings.vercel_ai_gateway_url,
                timeout=settings.jev_request_timeout,
            )

        self.model = settings.jev_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        reraise=True,
    )
    async def _call_primary(self, messages: list[dict], response_format: dict) -> dict:
        response = await self.primary_client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format=response_format,
            temperature=0.1,
            max_tokens=500,
        )
        return json.loads(response.choices[0].message.content)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        reraise=True,
    )
    async def _call_failover(self, messages: list[dict], response_format: dict) -> dict:
        if not self.failover_client:
            raise RuntimeError("Failover client not configured")
        response = await self.failover_client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format=response_format,
            temperature=0.1,
            max_tokens=500,
        )
        return json.loads(response.choices[0].message.content)

    async def score_article(self, context: JevPromptContext) -> JevScore:
        """Score a news article for relevance, direction, and confidence"""
        start_time = time.perf_counter()

        messages = self._build_messages(context)
        response_format = self._get_response_format()

        # Try primary, then failover
        last_error = None
        for _attempt, client_func in enumerate([
            ("primary", self._call_primary),
            ("failover", self._call_failover),
        ]):
            name, func = client_func
            try:
                result = await func(messages, response_format)
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                logger.info(f"Jev scoring succeeded via {name} in {latency_ms}ms")
                return self._parse_result(result, context, latency_ms)
            except Exception as e:
                last_error = e
                logger.warning(f"Jev {name} failed: {e}")
                if name == "failover":
                    break

        raise RuntimeError(f"All Jev endpoints failed. Last error: {last_error}")

    def _build_messages(self, context: JevPromptContext) -> list[dict]:
        system_prompt = """You are a financial news analyst for prediction markets.
Your task: assess how a news article affects the probability of a specific binary event.

Output a structured JSON with exactly these fields:
- relevance (0-1): How directly relevant is this article to the event?
- direction ("up" | "down" | "neutral"): Does this news increase, decrease, or not change the probability of "Yes"?
- confidence (0-1): How confident are you in this assessment?
- reasoning (string): Brief explanation (max 2 sentences)."""

        user_prompt = f"""Market: {context.market_title}
Description: {context.market_description}
Current Yes Price: {context.current_yes_price:.2%}

Article: {context.article_title}
Content: {context.article_content[:8000]}"""

        if context.article_summary:
            user_prompt += f"\n\nSummary: {context.article_summary}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _get_response_format(self) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "jev_score",
                "schema": {
                    "type": "object",
                    "properties": {
                        "relevance": {"type": "number", "minimum": 0, "maximum": 1},
                        "direction": {"type": "string", "enum": ["up", "down", "neutral"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reasoning": {"type": "string", "maxLength": 500},
                    },
                    "required": ["relevance", "direction", "confidence", "reasoning"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    def _parse_result(self, result: dict, context: JevPromptContext, latency_ms: int) -> JevScore:
        return JevScore(
            article_id=context.article_id if hasattr(context, 'article_id') else UUID(int=0),
            market_ticker=context.market_ticker,
            relevance=result["relevance"],
            direction=Direction(result["direction"]),
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            latency_ms=latency_ms,
        )


async def score_article(article: NewsArticle, market_ticker: str, market_title: str,
                        market_description: str, current_yes_price: float) -> JevScore:
    """Convenience function to score a single article"""
    client = JevClient()
    context = JevPromptContext(
        market_ticker=market_ticker,
        market_title=market_title,
        market_description=market_description,
        current_yes_price=current_yes_price,
        article_title=article.title,
        article_content=article.content,
        article_summary=article.summary,
    )
    # Attach article_id for the score
    context.article_id = article.id
    return await client.score_article(context)

