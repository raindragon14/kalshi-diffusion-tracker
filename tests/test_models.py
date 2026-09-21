import pytest
from src.models.market import KalshiMarket, PriceSnapshot, MarketStatus
from src.models.news import NewsArticle
from src.models.scoring import JevScore, Direction
from src.models.diffusion import DiffusionFit
from datetime import datetime, UTC
from uuid import uuid4


def test_market_model():
    market = KalshiMarket(
        ticker="FEDFUNDS",
        title="Fed Funds Rate > 4.5% by Dec 2024",
        category="economics",
        status=MarketStatus.OPEN,
        yes_bid=65,
        no_bid=35,
        open_time=datetime.now(UTC),
        close_time=datetime.now(UTC),
    )
    assert market.yes_probability == 0.65
    assert market.ticker == "FEDFUNDS"


def test_price_snapshot():
    snap = PriceSnapshot(
        ticker="FEDFUNDS",
        timestamp=datetime.now(UTC),
        yes_bid=65,
    )
    assert snap.yes_probability == 0.65


def test_jev_score():
    score = JevScore(
        article_id=uuid4(),
        market_ticker="FEDFUNDS",
        relevance=0.8,
        direction=Direction.UP,
        confidence=0.9,
        reasoning="Fed hawkish comments increase rate hike probability",
    )
    assert score.directional_impact == 0.8
    assert abs(score.weighted_confidence - 0.72) < 1e-10


def test_jev_score_down():
    score = JevScore(
        article_id=uuid4(),
        market_ticker="FEDFUNDS",
        relevance=0.6,
        direction=Direction.DOWN,
        confidence=0.7,
    )
    assert score.directional_impact == -0.6


def test_jev_score_neutral():
    score = JevScore(
        article_id=uuid4(),
        market_ticker="FEDFUNDS",
        relevance=0.3,
        direction=Direction.NEUTRAL,
        confidence=0.5,
    )
    assert score.directional_impact == 0.0


def test_news_article():
    article = NewsArticle(
        url="https://example.com/article",
        title="Test Article",
        content="Content here",
        published_at=datetime.now(UTC),
        source="test",
    )
    assert str(article.url) == "https://example.com/article"


def test_diffusion_fit():
    now = datetime.now(UTC)
    fit = DiffusionFit(
        score_id=uuid4(),
        article_id=uuid4(),
        market_ticker="FEDFUNDS",
        news_time=now,
        price_at_news=0.5,
        amplitude=0.15,
        timescale=1800.0,
        baseline=0.02,
        half_life=1247.0,
        r_squared=0.75,
        n_points=20,
        rmse=0.01,
        fit_start=now,
        fit_end=now,
    )
    assert abs(fit.initial_impact - 0.17) < 1e-10
    assert fit.is_significant is True


def test_diffusion_fit_not_significant():
    now = datetime.now(UTC)
    fit = DiffusionFit(
        score_id=uuid4(),
        article_id=uuid4(),
        market_ticker="FEDFUNDS",
        news_time=now,
        price_at_news=0.5,
        amplitude=0.005,
        timescale=1800.0,
        baseline=0.0,
        half_life=1247.0,
        r_squared=0.1,
        n_points=5,
        rmse=0.05,
        fit_start=now,
        fit_end=now,
    )
    assert fit.is_significant is False