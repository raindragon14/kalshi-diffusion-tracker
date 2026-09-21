from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class MarketStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    SETTLED = "settled"


class KalshiMarket(BaseModel):
    ticker: str
    title: str
    category: str
    status: MarketStatus
    yes_bid: int | None = None  # Price in cents (0-100)
    no_bid: int | None = None
    last_price: int | None = None
    volume: int | None = None
    open_time: datetime
    close_time: datetime
    settlement_time: datetime | None = None
    result: bool | None = None  # True = Yes, False = No

    @property
    def yes_probability(self) -> float | None:
        """Convert yes_bid from cents to probability 0-1"""
        if self.yes_bid is not None:
            return self.yes_bid / 100.0
        return None


class PriceSnapshot(BaseModel):
    ticker: str
    timestamp: datetime
    yes_bid: int | None = None
    no_bid: int | None = None
    last_price: int | None = None
    volume: int | None = None

    @property
    def yes_probability(self) -> float | None:
        if self.yes_bid is not None:
            return self.yes_bid / 100.0
        return None


class MarketsResponse(BaseModel):
    markets: list[KalshiMarket]
    cursor: str | None = None
