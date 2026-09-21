#!/usr/bin/env python3
"""Fit diffusion curves and evaluate calibration"""
import logging
from datetime import timedelta

from src.config import settings
from src.diffusion.calibration import evaluate_calibration, evaluate_directional_accuracy
from src.diffusion.fitter import build_price_series, fit_diffusion_curve
from src.storage.db import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_diffusion_fitting():
    """Fit diffusion curves for all scored articles"""
    scores = db.get_scores_for_market(settings.kalshi_market_ticker, limit=100)
    if not scores:
        logger.info("No scores to process")
        return

    logger.info(f"Processing diffusion for {len(scores)} scores")

    for score in scores:
        # Skip if already has a fit
        existing_fits = db.get_diffusion_fits(settings.kalshi_market_ticker, limit=1000)
        if any(f.score_id == score.id for f in existing_fits):
            continue

        # Get price snapshots around news time
        news_time = score.created_at  # Approximate - ideally use article.published_at
        article = db.get_article(score.article_id)
        if article:
            news_time = article.published_at

        price_snapshots = db.get_price_snapshots(
            settings.kalshi_market_ticker,
            since=news_time - timedelta(minutes=30),
            limit=500,
        )

        if len(price_snapshots) < 3:
            logger.warning(f"Insufficient price data for score {score.id}")
            continue

        # Find price at news time
        price_at_news = None
        for snap in price_snapshots:
            if snap.timestamp >= news_time and snap.yes_probability is not None:
                price_at_news = snap.yes_probability
                break

        if price_at_news is None:
            logger.warning(f"No price at news time for score {score.id}")
            continue

        # Build price series
        price_points = build_price_series(price_snapshots, news_time, price_at_news)

        if len(price_points) < 3:
            logger.warning(f"Insufficient price points after news for score {score.id}")
            continue

        # Fit curve
        fit = fit_diffusion_curve(
            price_points=price_points,
            score_id=score.id,
            article_id=score.article_id,
            market_ticker=score.market_ticker,
            news_time=news_time,
            price_at_news=price_at_news,
        )

        if fit and fit.is_significant:
            db.insert_diffusion_fit(fit)
            logger.info(f"  Fit saved: half-life={fit.half_life:.0f}s, R²={fit.r_squared:.3f}")
        else:
            logger.debug(f"  Fit not significant or failed for score {score.id}")


def run_calibration():
    """Evaluate calibration of Jev scores"""
    scores = db.get_scores_for_market(settings.kalshi_market_ticker, limit=100)
    if len(scores) < 5:
        logger.info(f"Insufficient scores for calibration: {len(scores)}")
        return

    # Get price changes after each score
    price_changes = []
    for score in scores:
        article = db.get_article(score.article_id)
        if not article:
            price_changes.append(0.0)
            continue

        # Get price at news time and after diffusion window
        snapshots = db.get_price_snapshots(
            settings.kalshi_market_ticker,
            since=article.published_at,
            limit=200,
        )

        price_at_news = None
        price_after = None
        for snap in snapshots:
            if snap.yes_probability is None:
                continue
            if price_at_news is None and snap.timestamp >= article.published_at:
                price_at_news = snap.yes_probability
            if snap.timestamp >= article.published_at + timedelta(seconds=settings.diffusion_window):
                price_after = snap.yes_probability
                break

        if price_at_news is not None and price_after is not None:
            price_changes.append(price_after - price_at_news)
        else:
            price_changes.append(0.0)

    # Evaluate directional accuracy
    realized, accuracy = evaluate_directional_accuracy(scores, price_changes)
    logger.info(f"Directional accuracy: {accuracy:.2%} ({sum(1 for r in realized if r != 0)}/{len(realized)} non-neutral)")

    # Evaluate calibration
    cal_result = evaluate_calibration(scores, realized, settings.kalshi_market_ticker)
    db.insert_calibration(cal_result)

    logger.info(f"Calibration: Brier={cal_result.brier_score:.4f}, ECE={cal_result.ece:.4f}, MCE={cal_result.mce:.4f}")


def main():
    run_diffusion_fitting()
    run_calibration()


if __name__ == "__main__":
    main()
