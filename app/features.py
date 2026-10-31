"""Student C serving-side feature assembly.

The formulas are identical to the training pipeline
(at3_crypto/modeling/student_c_tc.py in the experiments repository):
`projected_trend_score` and the derived ratios are copied verbatim, and the
CONTRACT.md section 4 features come from the shared `app.core` modules.
"""

from __future__ import annotations

import math
from typing import Any, Final

import numpy as np
import pandas as pd

from app.core.engineered_features import engineer_features
from app.core.targets import FORECAST_HORIZON, TC_LABELS, classify_tc, trend_score

# Features this service OWNS and publishes on POST /features (CONTRACT.md 2/4).
OWNED_FEATURES: Final[list[str]] = ["Volatility20", "MaxPrice20", "MaxDrawdown", "GapRisk"]
# Extra feature engineered by Student C, published alongside (extra keys are allowed).
EXTRA_OWNED_FEATURES: Final[list[str]] = ["TrendScoreProj7"]

OWNED_FEATURE_DEFINITIONS: Final[dict[str, str]] = {
    "Volatility20": "STD(DailyReturn_t ... DailyReturn_t-19), sample std (ddof=1), not annualised",
    "MaxPrice20": "MAX(High_t ... High_t-19)",
    "MaxDrawdown": "(MaxPrice20 - Close_t) / MaxPrice20",
    "GapRisk": "MEAN(|Open_t - Close_t-1| / Close_t-1) over 20 days",
    "TrendScoreProj7": "TrendScore at t+7 if Close stays at Close_t for 7 days: "
    "0.7*(Close_t / ((sum(C_t-12..C_t) + 7*C_t)/20) - 1) + 0.3*(Close_t / C_t-13 - 1)",
}

# Peer features this model consumes, by owner (CONTRACT.md section 2).
PEER_FEATURES: Final[dict[str, list[str]]] = {
    "student_a": ["Return20", "Return5"],
    "student_b": ["MA5", "MA20"],
}


def projected_trend_score(frame: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.Series:
    """TrendScore at t+horizon assuming Close stays flat at Close_t (Student C)."""
    known = 20 - horizon
    grouped = frame.groupby("token", sort=False)["close"]
    close = frame["close"]
    known_sum = grouped.transform(lambda s: s.rolling(known, min_periods=known).sum())
    ma20_projected = (known_sum + horizon * close) / 20
    return20_projected = close / grouped.shift(known) - 1
    return 0.7 * (close / ma20_projected - 1) + 0.3 * return20_projected


def compute_local_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """All CONTRACT.md features + Student C's projection, per day, oldest first."""
    frame = engineer_features(ohlcv).sort_values(["token", "date"]).reset_index(drop=True)
    frame["TrendScoreProj7"] = projected_trend_score(frame)
    frame["TrendScore"] = trend_score(frame)
    frame["TC"] = classify_tc(frame["TrendScore"])
    return frame


def build_model_row(
    feature_columns: list[str], peer_values: dict[str, float | None], local_row: pd.Series
) -> pd.DataFrame:
    """One-row model input. Peer values that are missing stay NaN.

    Derived inputs follow the training definitions and are NaN whenever an
    ingredient is missing, exactly as in the training-time masking.
    """
    def val(name: str) -> float:
        value = peer_values.get(name)
        return float(value) if value is not None and math.isfinite(value) else np.nan

    ma5, ma20, r20, r5 = val("MA5"), val("MA20"), val("Return20"), val("Return5")
    ratio = ma5 / ma20 if np.isfinite(ma5) and np.isfinite(ma20) and ma20 != 0 else np.nan
    close = float(local_row["close"])
    values = {
        "MA5": ma5,
        "MA20": ma20,
        "Return20": r20,
        "Return5": r5,
        "MA5_MA20_Ratio": ratio,
        "MA5_to_Close": ma5 / close if np.isfinite(ma5) else np.nan,
        "MA20_to_Close": ma20 / close if np.isfinite(ma20) else np.nan,
        "TrendScore": 0.7 * (ratio - 1) + 0.3 * r20 if np.isfinite(ratio) and np.isfinite(r20) else np.nan,
        "TrendScoreProj7": float(local_row["TrendScoreProj7"]),
        "Volatility20": float(local_row["Volatility20"]),
        "MaxDrawdown": float(local_row["MaxDrawdown"]),
        "GapRisk": float(local_row["GapRisk"]),
    }
    return pd.DataFrame([[values[c] for c in feature_columns]], columns=feature_columns, dtype="float64")


def json_number(value: Any, digits: int | None = None) -> float | None:
    """Float for JSON; NaN/inf -> None (NaN is invalid JSON, CONTRACT.md 6.3).

    Full precision by default: peers consume these values as model inputs, and
    small features such as GapRisk (~3e-4) would lose accuracy if rounded.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits) if digits is not None else number


def label_of(class_id: Any) -> str | None:
    if class_id is None or (isinstance(class_id, float) and math.isnan(class_id)):
        return None
    return TC_LABELS[int(class_id)]
