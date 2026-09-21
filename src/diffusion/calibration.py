import logging

import numpy as np

from src.models.diffusion import CalibrationResult
from src.models.scoring import JevScore

logger = logging.getLogger(__name__)


def compute_brier_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute Brier score for binary predictions"""
    return np.mean((predictions - outcomes) ** 2)


def brier_decomposition(predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> dict:
    """Decompose Brier score into reliability, resolution, uncertainty"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    reliability = 0.0
    resolution = 0.0
    overall_mean = np.mean(outcomes)

    for i in range(n_bins):
        mask = bin_indices == i
        if not np.any(mask):
            continue
        bin_pred = np.mean(predictions[mask])
        bin_outcome = np.mean(outcomes[mask])
        bin_count = np.sum(mask)

        reliability += bin_count * (bin_pred - bin_outcome) ** 2
        resolution += bin_count * (bin_outcome - overall_mean) ** 2

    n = len(predictions)
    reliability /= n
    resolution /= n
    uncertainty = overall_mean * (1 - overall_mean)

    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
    }


def compute_reliability_diagram(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Compute reliability diagram data points"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bins = []
    for i in range(n_bins):
        mask = bin_indices == i
        count = np.sum(mask)
        if count == 0:
            continue

        bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
        bin_confidence = np.mean(predictions[mask])
        bin_accuracy = np.mean(outcomes[mask])

        bins.append({
            "bin_center": float(bin_center),
            "bin_low": float(bin_edges[i]),
            "bin_high": float(bin_edges[i + 1]),
            "confidence": float(bin_confidence),
            "accuracy": float(bin_accuracy),
            "count": int(count),
        })

    return bins


def expected_calibration_error(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE)"""
    bins = compute_reliability_diagram(predictions, outcomes, n_bins)
    total = sum(b["count"] for b in bins)
    if total == 0:
        return 0.0
    ece = sum(b["count"] * abs(b["confidence"] - b["accuracy"]) for b in bins) / total
    return ece


def maximum_calibration_error(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Maximum Calibration Error (MCE)"""
    bins = compute_reliability_diagram(predictions, outcomes, n_bins)
    if not bins:
        return 0.0
    return max(abs(b["confidence"] - b["accuracy"]) for b in bins)


def evaluate_calibration(
    scores: list[JevScore],
    realized_directions: list[int],  # 1 for up, -1 for down, 0 for neutral
    market_ticker: str,
) -> CalibrationResult:
    """Evaluate calibration of Jev scores against realized outcomes"""
    if len(scores) != len(realized_directions):
        raise ValueError("Scores and outcomes must have same length")

    # For calibration, we use confidence as predicted probability
    # and check if high confidence correlates with correct directional predictions
    predictions = []
    outcomes = []

    for score, realized in zip(scores, realized_directions):
        # Predicted probability that direction is correct
        # We use weighted_confidence as the prediction
        pred = score.weighted_confidence

        # Outcome: 1 if direction matches (or neutral and realized neutral), else 0
        predicted_direction = 1 if score.direction.value == "up" else (-1 if score.direction.value == "down" else 0)
        outcome = 1 if predicted_direction == realized else 0

        predictions.append(pred)
        outcomes.append(outcome)

    predictions = np.array(predictions)
    outcomes = np.array(outcomes)

    brier = compute_brier_score(predictions, outcomes)
    decomposition = brier_decomposition(predictions, outcomes)
    reliability_bins = compute_reliability_diagram(predictions, outcomes)
    ece = expected_calibration_error(predictions, outcomes)
    mce = maximum_calibration_error(predictions, outcomes)

    return CalibrationResult(
        n_samples=len(scores),
        brier_score=float(brier),
        brier_score_decomposition=decomposition,
        reliability_bins=reliability_bins,
        ece=float(ece),
        mce=float(mce),
        market_ticker=market_ticker,
    )


def evaluate_directional_accuracy(
    scores: list[JevScore],
    price_changes: list[float],  # Actual price changes after news
    threshold: float = 0.01,
) -> tuple[list[int], float]:
    """Convert price changes to realized directions and compute accuracy"""
    realized = []
    for change in price_changes:
        if change > threshold:
            realized.append(1)  # up
        elif change < -threshold:
            realized.append(-1)  # down
        else:
            realized.append(0)  # neutral

    # Compute accuracy
    correct = 0
    for score, r in zip(scores, realized):
        pred = 1 if score.direction.value == "up" else (-1 if score.direction.value == "down" else 0)
        if pred == r:
            correct += 1

    accuracy = correct / len(scores) if scores else 0.0
    return realized, accuracy
