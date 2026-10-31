"""Student C - Trend Category (TC) API.  Group 08, 36120 AT3 (Spring 2026).

Forecasts the Trend Category of a cryptocurrency 7 days ahead with a tuned
XGBoost Classifier, and publishes Student C's engineered features
(Volatility20, MaxPrice20, MaxDrawdown, GapRisk) for the other services.

Endpoints (CONTRACT.md section 6):
    GET  /                 API metadata and documentation
    GET  /health           health check
    POST /features         Student C's engineered features (+ history for charts)
    POST /predict          7-day-ahead Trend Category with confidence
    GET  /model-metadata   target, algorithm, features, importance, labels, metrics

Run locally:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import peers
from app.core.kraken import SUPPORTED_TOKENS, KrakenClient, KrakenError
from app.core.targets import FORECAST_HORIZON, TC_LABELS, classify_tc
from app.features import (
    EXTRA_OWNED_FEATURES,
    OWNED_FEATURE_DEFINITIONS,
    OWNED_FEATURES,
    PEER_FEATURES,
    build_model_row,
    compute_local_features,
    json_number,
    label_of,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("student_c_api")

API_VERSION: Final = "1.0.0"
ROOT: Final = Path(__file__).resolve().parents[1]
MODEL_PATH: Final = ROOT / "models" / "tc_xgboost.joblib"
METRICS_PATH: Final = ROOT / "models" / "metrics.json"
GITHUB_URL: Final = os.getenv(
    "GITHUB_REPO_URL", "https://github.com/<github-username>/36120-26SP-AT3-Group08-StudentC-api"
)
SUPPORTED: Final[list[str]] = list(SUPPORTED_TOKENS)
MAX_HISTORY_DAYS: Final = 365


# ----------------------------------------------------------------------------
# Errors - one shape everywhere (CONTRACT.md 6.6)
# ----------------------------------------------------------------------------


class ApiError(Exception):
    def __init__(self, status: int, body: dict[str, Any]):
        self.status, self.body = status, body


def unsupported_token() -> ApiError:
    return ApiError(400, {"error": "Unsupported cryptocurrency", "supported_tokens": SUPPORTED})


def unsupported_horizon() -> ApiError:
    return ApiError(422, {"error": "Unsupported forecast_horizon", "supported_horizons": [FORECAST_HORIZON]})


def validate(token: str, horizon: int) -> str:
    token = token.strip().upper()
    if token not in SUPPORTED_TOKENS:
        raise unsupported_token()
    if horizon != FORECAST_HORIZON:
        raise unsupported_horizon()
    return token


# ----------------------------------------------------------------------------
# Model and market data
# ----------------------------------------------------------------------------


def load_bundle() -> dict[str, Any]:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    if not isinstance(model, xgb.XGBClassifier):  # defensive: fall back to native format
        model = xgb.XGBClassifier()
        model.load_model(ROOT / "models" / "tc_xgboost.json")
        bundle["model"] = model
    bundle["metrics_json"] = json.loads(METRICS_PATH.read_text())
    logger.info("Loaded model %s with features %s", bundle["model_version"], bundle["feature_columns"])
    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_bundle()
    # Replaceable in tests by any object with fetch_ohlcv(token) -> DataFrame.
    app.state.market = KrakenClient()
    yield


app = FastAPI(
    title="Trend Category API (Student C, Group 08)",
    description="Forecasts the Trend Category of a cryptocurrency 7 days ahead (XGBoost Classifier) "
    "and serves Student C's engineered features.",
    version=API_VERSION,
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(ApiError)
async def _api_error(_: Request, exc: ApiError):
    return JSONResponse(status_code=exc.status, content=exc.body)


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError):
    details = [
        {"field": ".".join(str(p) for p in e.get("loc", []) if p != "body"), "message": e.get("msg")}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"error": "Invalid request", "details": details})


@app.exception_handler(Exception)
async def _unexpected(_: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


async def local_features(token: str) -> pd.DataFrame:
    """Daily features for `token` from Kraken (complete days only), oldest first."""
    try:
        ohlcv = await run_in_threadpool(app.state.market.fetch_ohlcv, token)
    except KrakenError as exc:
        logger.warning("Kraken unavailable for %s: %s", token, exc)
        raise ApiError(503, {"error": "Market data source unavailable", "retry_after_seconds": 60}) from exc
    return compute_local_features(ohlcv)


# ----------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------


class FeaturesRequest(BaseModel):
    token: str = Field(..., examples=["BTC"])
    forecast_horizon: int = Field(FORECAST_HORIZON, examples=[7])
    history_days: int = Field(1, ge=1, le=MAX_HISTORY_DAYS, examples=[1])


class PredictRequest(BaseModel):
    token: str = Field(..., examples=["BTC"])
    forecast_horizon: int = Field(FORECAST_HORIZON, examples=[7])


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "project": "Trend Category API",
        "description": "Predicts the Trend Category (TC) of cryptocurrencies 7 days into the future: "
        "Strong Downtrend, Downtrend, Sideways, Uptrend or Strong Uptrend, from "
        "TrendScore = 0.7*((MA5/MA20) - 1) + 0.3*Return20. Also serves Student C's engineered "
        "features (Volatility20, MaxPrice20, MaxDrawdown, GapRisk) to the other Group 08 services.",
        "algorithm": "XGBoost Classifier",
        "version": API_VERSION,
        "github": GITHUB_URL,
        "group": "Group 08 - 36120 Advanced Machine Learning Application, Spring 2026",
        "student": {"role": "Student C", "name": "Nandeeswari Kumaravelu", "id": "25980436"},
        "forecast_horizon": FORECAST_HORIZON,
        "supported_tokens": SUPPORTED,
        "data_source": "Kraken public OHLC API (daily USD candles, incomplete current day dropped)",
        "endpoints": ["/", "/health", "/features", "/predict", "/model-metadata"],
        "endpoint_details": {
            "GET /": "this document",
            "GET /health": "service and model status",
            "POST /features": 'body {"token": "BTC", "forecast_horizon": 7, "history_days": 1}',
            "POST /predict": 'body {"token": "BTC", "forecast_horizon": 7}',
            "GET /model-metadata": "target, algorithm, features, feature importance, labels, metrics",
            "GET /docs": "interactive Swagger UI",
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    bundle = getattr(app.state, "bundle", None)
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "model_version": bundle["model_version"] if bundle else None,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "peers_configured": {p: url is not None for p, url in peers.configured_peers().items()},
    }


@app.post("/features")
async def features(request: FeaturesRequest) -> dict[str, Any]:
    token = validate(request.token, request.forecast_horizon)
    frame = await local_features(token)
    published = OWNED_FEATURES + EXTRA_OWNED_FEATURES
    latest = frame.iloc[-1]
    history = frame.tail(request.history_days)
    return {
        "token": token,
        "as_of_date": str(latest["date"].date()),
        "source": "kraken",
        "features": {name: json_number(latest[name]) for name in published},
        "definitions": OWNED_FEATURE_DEFINITIONS,
        "history": [
            {
                "date": str(row["date"].date()),
                "close": json_number(row["close"]),
                "high": json_number(row["high"]),
                **{name: json_number(row[name]) for name in published},
            }
            for _, row in history.iterrows()
        ],
    }


@app.post("/predict")
async def predict(request: PredictRequest) -> dict[str, Any]:
    token = validate(request.token, request.forecast_horizon)
    bundle = app.state.bundle
    model: xgb.XGBClassifier = bundle["model"]
    columns: list[str] = bundle["feature_columns"]

    frame = await local_features(token)
    latest = frame.iloc[-1]
    as_of = latest["date"].date()

    # Peer-owned features from the peers' APIs (in parallel); missing -> NaN.
    peer_results = await peers.fetch_peer_features(token)
    peer_values: dict[str, float | None] = {}
    feature_sources: dict[str, str] = {}
    for peer, result in peer_results.items():
        for name in PEER_FEATURES[peer]:
            value = result.values.get(name) if result.ok else None
            peer_values[name] = value
            if value is not None:
                feature_sources[name] = f"{peer}_api" + (" (cached)" if result.cached else "")
    for name in ("TrendScoreProj7", "Volatility20", "MaxDrawdown", "GapRisk"):
        feature_sources[name] = "student_c (local, Kraken)"

    x = build_model_row(columns, peer_values, latest)
    missing = [c for c in columns if np.isnan(x.iloc[0][c])]

    probabilities = model.predict_proba(x)[0]
    class_id = int(np.argmax(probabilities))

    # Per-prediction explanation: SHAP contributions (XGBoost pred_contribs) for the
    # predicted class, in log-odds units. The last column is the bias term.
    # Multiclass shape: (rows, classes, features + 1).
    contribs = model.get_booster().predict(xgb.DMatrix(x, missing=np.nan), pred_contribs=True)
    class_contribs = np.asarray(contribs)[0, class_id, :-1]
    explanation = sorted(
        ({"feature": c, "value": json_number(x.iloc[0][c], 6), "contribution": json_number(v, 4)}
         for c, v in zip(columns, class_contribs)),
        key=lambda d: -abs(d["contribution"] or 0),
    )

    return {
        "prediction": TC_LABELS[class_id],
        "confidence": round(float(probabilities[class_id]), 4),
        "probabilities": {label: round(float(p), 4) for label, p in zip(TC_LABELS, probabilities)},
        "forecast_horizon": FORECAST_HORIZON,
        "prediction_date": str(as_of + timedelta(days=FORECAST_HORIZON)),
        "token": token,
        "as_of_date": str(as_of),
        "model_version": bundle["model_version"],
        "missing_features": missing,
        "feature_sources": feature_sources,
        "peer_status": {
            peer: ("ok" + (" (cached)" if r.cached else "")) if r.ok else f"unavailable: {r.error}"
            for peer, r in peer_results.items()
        },
        "features_used": {c: json_number(x.iloc[0][c], 6) for c in columns},
        "explanation": explanation,
        # Context for the app: today's trend from the same formulas, computed from Kraken.
        "current": {
            "date": str(as_of),
            "close": json_number(latest["close"]),
            "trend": label_of(latest["TC"]),
            "trend_score": json_number(latest["TrendScore"], 6),
            "projected_trend_score_t7": json_number(latest["TrendScoreProj7"], 6),
            "projected_trend_if_flat": label_of(classify_tc(json_number(latest["TrendScoreProj7"]))),
            "MA5": json_number(latest["MA5"]),
            "MA20": json_number(latest["MA20"]),
            "Return20": json_number(latest["Return20"], 6),
            "Volatility20": json_number(latest["Volatility20"], 6),
        },
    }


@app.get("/model-metadata")
def model_metadata() -> dict[str, Any]:
    m = app.state.bundle["metrics_json"]
    return {
        "target": m["target"],
        "target_description": m["target_description"],
        "algorithm": m["algorithm"],
        "forecast_horizon": m["forecast_horizon"],
        "features": m["features"],
        "feature_importance": m["feature_importance"],
        "feature_sources": m["feature_sources"],
        "classification_labels": m["classification_labels"],
        "classification_rules": m["classification_rules"],
        "metrics": m["metrics"],
        "baselines": {k: v["macro_f1"] for k, v in m["baselines"].items()},
        "per_class_recall": m["per_class_recall"],
        "per_token_macro_f1": {k: v["macro_f1"] for k, v in m["per_token"].items()},
        "degradation_macro_f1": {k: v["macro_f1"] for k, v in m["degradation"].items()},
        "confusion_matrix": m["confusion_matrix"],
        "training_period": m["training_period"],
        "best_params": m["best_params"],
        "cv_strategy": m["cv_strategy"],
        "n_trials": m["n_trials"],
        "supported_tokens": m["supported_tokens"],
        "owned_features": OWNED_FEATURES,
        "model_version": m["model_version"],
        "created_at": m["created_at"],
    }
