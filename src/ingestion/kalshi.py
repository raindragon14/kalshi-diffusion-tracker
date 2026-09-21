import asyncio
import logging
from datetime import datetime

import httpx

from src.config import settings
from src.models.market import KalshiMarket, PriceSnapshot

logger = logging.getLogger(__name__)


class KalshiClient:
    def __init__(self):
        self.base_url = settings.kalshi_base_url
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
            headers={"Accept": "application/json"},
        )

    async def close(self):
        await self.client.aclose()

    async def get_markets(self, limit: int = 100, status: str = "open") -> list[KalshiMarket]:
        """Fetch all markets, optionally filtered by status"""
        params = {"limit": limit, "status": status}
        response = await self.client.get("/markets", params=params)
        response.raise_for_status()
        data = response.json()
        return [KalshiMarket(**m) for m in data.get("markets", [])]

    async def get_market(self, ticker: str) -> KalshiMarket | None:
        """Fetch a specific market by ticker"""
        try:
            response = await self.client.get(f"/markets/{ticker}")
            response.raise_for_status()
            return KalshiMarket(**response.json().get("market", {}))
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def get_price_snapshot(self, ticker: str) -> PriceSnapshot | None:
        """Get current price snapshot for a market"""
        market = await self.get_market(ticker)
        if not market:
            return None

        return PriceSnapshot(
            ticker=ticker,
            timestamp=datetime.utcnow(),
            yes_bid=market.yes_bid,
            no_bid=market.no_bid,
            last_price=market.last_price,
            volume=market.volume,
        )

    async def poll_prices(self, ticker: str, interval: int, callback, stop_event: asyncio.Event):
        """Continuously poll prices and call callback with each snapshot"""
        logger.info(f"Starting price polling for {ticker} every {interval}s")
        while not stop_event.is_set():
            try:
                snapshot = await self.get_price_snapshot(ticker)
                if snapshot:
                    await callback(snapshot)
                else:
                    logger.warning(f"No price data for {ticker}")
            except Exception as e:
                logger.error(f"Error polling price for {ticker}: {e}")

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except TimeoutError:
                continue  # Normal timeout, continue polling

        logger.info(f"Stopped price polling for {ticker}")
