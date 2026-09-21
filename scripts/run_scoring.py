#!/usr/bin/env python3
"""Score unscored articles with Jev"""
import asyncio
import logging

from src.config import settings
from src.scoring.jev_client import score_article
from src.storage.db import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def score_all_unscored():
    """Score all unscored articles for the configured market"""
    market = db.get_market(settings.kalshi_market_ticker)
    if not market:
        logger.error(f"Market {settings.kalshi_market_ticker} not found in DB")
        return

    articles = db.get_unscored_articles(limit=50)
    if not articles:
        logger.info("No unscored articles found")
        return

    logger.info(f"Scoring {len(articles)} articles for {market.ticker}")

    for article in articles:
        try:
            # Get price at article publication time (or latest)
            price_snapshots = db.get_price_snapshots(
                market.ticker,
                since=article.published_at,
                limit=1,
            )
            current_price = price_snapshots[0].yes_probability if price_snapshots else (market.yes_probability or 0.5)

            logger.info(f"Scoring: {article.title[:80]}...")

            score = await score_article(
                article=article,
                market_ticker=market.ticker,
                market_title=market.title,
                market_description=f"Prediction market on {market.title}",
                current_yes_price=current_price,
            )

            db.insert_score(score)
            db.mark_article_scored(article.id, score.id)
            logger.info(f"  -> relevance={score.relevance:.2f}, direction={score.direction.value}, confidence={score.confidence:.2f}")

        except Exception as e:
            logger.error(f"Failed to score article {article.id}: {e}")


async def main():
    await score_all_unscored()


if __name__ == "__main__":
    asyncio.run(main())
