import logging
from datetime import datetime
from uuid import UUID

import numpy as np
from scipy.optimize import curve_fit

from src.config import settings
from src.models.diffusion import DiffusionFit, PriceSeriesPoint

logger = logging.getLogger(__name__)


def exponential_decay(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    """Exponential decay model: A * exp(-t / tau) + C"""
    return A * np.exp(-t / tau) + C


def fit_diffusion_curve(
    price_points: list[PriceSeriesPoint],
    score_id: UUID,
    article_id: UUID,
    market_ticker: str,
    news_time: datetime,
    price_at_news: float,
) -> DiffusionFit | None:
    """Fit exponential decay to price movement after news event"""
    if len(price_points) < 3:
        logger.warning(f"Insufficient points for curve fitting: {len(price_points)}")
        return None

    # Extract time and price change arrays
    t = np.array([p.time_since_news for p in price_points])
    y = np.array([p.price_change for p in price_points])

    # Initial guess: amplitude ~ max change, tau ~ 1/3 of window, baseline ~ final value
    max_change = np.max(np.abs(y))
    initial_guess = [
        np.max(y) if np.max(y) > 0 else -np.max(np.abs(y)),  # A
        settings.diffusion_window / 3,  # tau
        y[-1],  # C
    ]

    # Bounds: A in [-1, 1], tau in [60, window*2], C in [-1, 1]
    bounds = (
        [-1.0, 60.0, -1.0],
        [1.0, settings.diffusion_window * 2, 1.0],
    )

    try:
        popt, pcov = curve_fit(
            exponential_decay,
            t,
            y,
            p0=initial_guess,
            bounds=bounds,
            maxfev=5000,
        )

        A, tau, C = popt

        # Calculate fit quality
        y_pred = exponential_decay(t, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        # Half-life
        half_life = tau * np.log(2) if tau > 0 else 0

        return DiffusionFit(
            score_id=score_id,
            article_id=article_id,
            market_ticker=market_ticker,
            news_time=news_time,
            price_at_news=price_at_news,
            amplitude=float(A),
            timescale=float(tau),
            baseline=float(C),
            half_life=float(half_life),
            r_squared=float(max(0.0, min(1.0, r_squared))),
            n_points=len(price_points),
            rmse=float(rmse),
            fit_start=price_points[0].timestamp,
            fit_end=price_points[-1].timestamp,
        )

    except Exception as e:
        logger.error(f"Curve fitting failed: {e}")
        return None


def build_price_series(
    price_snapshots: list,
    news_time: datetime,
    price_at_news: float,
    window: int | None = None,
) -> list[PriceSeriesPoint]:
    """Build price series points from snapshots within window"""
    if window is None:
        window = settings.diffusion_window

    points = []
    for snap in price_snapshots:
        if snap.timestamp < news_time:
            continue
        dt = (snap.timestamp - news_time).total_seconds()
        if dt > window:
            continue
        if snap.yes_probability is None:
            continue

        points.append(PriceSeriesPoint(
            timestamp=snap.timestamp,
            yes_probability=snap.yes_probability,
            time_since_news=dt,
            price_change=snap.yes_probability - price_at_news,
        ))

    return points
