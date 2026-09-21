from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DiffusionFit(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    score_id: UUID
    article_id: UUID
    market_ticker: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Event timing
    news_time: datetime
    price_at_news: float  # Yes probability at news time

    # Exponential decay fit: price_change = A * exp(-t / tau) + C
    # where t is time since news
    amplitude: float = Field(description="A - initial price impact magnitude")
    timescale: float = Field(description="tau - decay time constant (seconds)")
    baseline: float = Field(description="C - long-term baseline offset")
    half_life: float = Field(description="Time for impact to halve (seconds)")

    # Fit quality
    r_squared: float = Field(ge=0.0, le=1.0, description="Goodness of fit")
    n_points: int = Field(description="Number of price points used in fit")
    rmse: float = Field(description="Root mean squared error")

    # Time range of fit
    fit_start: datetime
    fit_end: datetime

    # Computed
    @property
    def initial_impact(self) -> float:
        """Total initial impact: A + C"""
        return self.amplitude + self.baseline

    @property
    def is_significant(self) -> bool:
        """Whether the diffusion signal is statistically meaningful"""
        return self.r_squared > 0.3 and abs(self.amplitude) > 0.01


class CalibrationResult(BaseModel):
    """Results from calibration evaluation"""
    n_samples: int
    brier_score: float
    brier_score_decomposition: dict  # {"reliability": x, "resolution": x, "uncertainty": x}
    reliability_bins: list[dict]  # [{"bin_center": 0.1, "confidence": 0.1, "accuracy": 0.15, "count": 10}, ...]
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error

    # Metadata
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    market_ticker: str
    model_version: str = "jev-1.13"


class PriceSeriesPoint(BaseModel):
    """Single price point for diffusion analysis"""
    timestamp: datetime
    yes_probability: float
    time_since_news: float  # seconds
    price_change: float  # from price_at_news
