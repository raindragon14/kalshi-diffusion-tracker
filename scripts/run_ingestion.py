#!/usr/bin/env python3
"""Run price and news ingestion"""
import asyncio
import logging
import signal

from src.config import settings
from src.ingestion.firecrawl import FirecrawlClient
from src.ingestion.kalshi import KalshiClient
from src.models.market import PriceSnapshot
from src.models.news import NewsArticle
from src.storage.db import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class IngestionRunner:
    def __init__(self):
        self.kalshi = KalshiClient()
        self.firecrawl = FirecrawlClient()
        self.stop_event = asyncio.Event()

    async def handle_price_snapshot(self, snapshot: PriceSnapshot):
        """Callback for each price snapshot"""
        db.insert_price_snapshot(snapshot)
        logger.debug(f"Stored price: {snapshot.ticker} @ {snapshot.yes_probability:.4f}")

    async def handle_new_article(self, article: NewsArticle):
        """Callback for each new article"""
        is_new = db.insert_article(article)
        if is_new:
            logger.info(f"New article: {article.title[:80]}...")
        else:
            logger.debug(f"Duplicate article: {article.title[:80]}...")

    async def run(self):
        # Ensure market exists in DB
        market = await self.kalshi.get_market(settings.kalshi_market_ticker)
        if market:
            db.upsert_market(market)
            logger.info(f"Tracking market: {market.ticker} - {market.title}")

        # Start polling tasks
        tasks = [
            asyncio.create_task(self.kalshi.poll_prices(
                settings.kalshi_market_ticker,
                settings.kalshi_poll_interval,
                self.handle_price_snapshot,
                self.stop_event,
            )),
            asyncio.create_task(self.firecrawl.poll_news(
                settings.firecrawl_search_queries,
                settings.firecrawl_poll_interval,
                self.handle_new_article,
                self.stop_event,
            )),
        ]

        # Wait for shutdown
        await asyncio.gather(*tasks, return_exceptions=True)

    async def shutdown(self):
        logger.info("Shutting down ingestion...")
        self.stop_event.set()
        await self.kalshi.close()
        await self.firecrawl.close()


async def main():
    runner = IngestionRunner()

    def signal_handler():
        asyncio.create_task(runner.shutdown())

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await runner.run()
    except asyncio.CancelledError:
        pass
    finally:
        await runner.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
