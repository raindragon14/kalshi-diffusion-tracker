import asyncio
import logging
from datetime import datetime

import httpx

from src.config import settings
from src.models.news import NewsArticle, SearchResult

logger = logging.getLogger(__name__)


class FirecrawlClient:
    def __init__(self):
        self.api_key = settings.firecrawl_api_key
        self.base_url = settings.firecrawl_base_url
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=60.0,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def close(self):
        await self.client.aclose()

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search for articles matching query"""
        payload = {
            "query": query,
            "limit": limit,
            "scrapeOptions": {"formats": ["markdown"]},
        }
        response = await self.client.post("/search", json=payload)
        response.raise_for_status()
        data = response.json()

        results = [
            SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                snippet=item.get("description", "") or item.get("snippet", ""),
                published_at=self._parse_date(item.get("publishedAt")),
                source=item.get("source", "unknown"),
            )
            for item in data.get("data", [])
        ]
        return results

    async def scrape(self, url: str) -> NewsArticle | None:
        """Scrape full article content from URL"""
        payload = {
            "url": url,
            "formats": ["markdown", "html"],
            "onlyMainContent": True,
        }
        try:
            response = await self.client.post("/scrape", json=payload)
            response.raise_for_status()
            data = response.json().get("data", {})

            return NewsArticle(
                url=data.get("url", url),
                title=data.get("metadata", {}).get("title", "Untitled"),
                content=data.get("markdown", "") or data.get("html", ""),
                summary=data.get("metadata", {}).get("description"),
                published_at=self._parse_date(data.get("metadata", {}).get("publishedAt")),
                source=data.get("metadata", {}).get("source", "firecrawl"),
                author=data.get("metadata", {}).get("author"),
                metadata=data.get("metadata", {}),
            )
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return None

    async def search_and_scrape(self, query: str, limit: int = 10) -> list[NewsArticle]:
        """Search and scrape in one go"""
        results = await self.search(query, limit)
        articles = []
        for result in results:
            article = await self.scrape(str(result.url))
            if article:
                articles.append(article)
        return articles

    def _parse_date(self, date_str: str | None) -> datetime | None:
        if not date_str:
            return None
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        logger.warning(f"Could not parse date: {date_str}")
        return None

    async def poll_news(self, queries: list[str], interval: int, callback, stop_event: asyncio.Event):
        """Continuously poll for new articles"""
        logger.info(f"Starting news polling for {len(queries)} queries every {interval}s")
        seen_urls = set()

        while not stop_event.is_set():
            for query in queries:
                try:
                    articles = await self.search_and_scrape(query, settings.firecrawl_max_articles)
                    for article in articles:
                        url_str = str(article.url)
                        if url_str not in seen_urls:
                            seen_urls.add(url_str)
                            await callback(article)
                except Exception as e:
                    logger.error(f"Error polling news for query '{query}': {e}")

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except TimeoutError:
                continue

        logger.info("Stopped news polling")
