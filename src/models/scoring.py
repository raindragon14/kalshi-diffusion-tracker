from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Direction(StrEnum):
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


class JevScore(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    article_id: UUID
    market_ticker: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Core scores from Jev
    relevance: float = Field(ge=0.0, le=1.0, description="How relevant is this news to the market (0-1)")
    direction: Direction = Field(description="Expected impact on Yes probability")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence in the assessment (0-1)")

    # Reasoning (optional but valuable)
    reasoning: str = Field(default="", description="Brief explanation of the assessment")

    # Metadata
    model_version: str = "jev-1.13"
    prompt_version: str = "1.0"
    latency_ms: int = 0

    # Computed fields
    @property
    def directional_impact(self) -> float:
        """Signed impact: +relevance for up, -relevance for down, 0 for neutral"""
        if self.direction == Direction.UP:
            return self.relevance
        if self.direction == Direction.DOWN:
            return -self.relevance
        return 0.0

    @property
    def weighted_confidence(self) -> float:
        """Confidence weighted by relevance"""
        return self.confidence * self.relevance


class JevPromptContext(BaseModel):
    market_ticker: str
    market_title: str
    market_description: str
    current_yes_price: float
    article_title: str
    article_content: str
    article_summary: str | None = None
