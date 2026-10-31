"""Offline tests for the Student C API.

Kraken is replaced by a fake client serving the committed snapshot (BTC, DOGE),
and the peer services are replaced by an httpx MockTransport that computes
their features from the same snapshot with the shared group formulas. No
network access is needed:  pytest -q
"""

from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main, peers
from app.core.engineered_features import engineer_features
from app.core.kraken import KrakenError
from app.core.targets import TC_LABELS, classify_tc
from app.features import compute_local_features

FIXTURE = Path(__file__).parent / "fixtures" / "kraken_btc_doge.csv"


class FakeKraken:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.data = pd.read_csv(FIXTURE)
        self.data["date"] = pd.to_datetime(self.data["date"], utc=True)

    def fetch_ohlcv(self, token: str, drop_incomplete: bool = True) -> pd.DataFrame:
        if self.fail:
            raise KrakenError("simulated outage")
        frame = self.data[self.data["token"] == token]
        if frame.empty:
            raise KrakenError(f"no data for {token}")
        return frame.reset_index(drop=True)


def peer_handler(down: set[str] = frozenset()):
    """Mock POST /features for Student A (Return20, Return5) and B (MA5, MA20)."""
    feats = engineer_features(FakeKraken().data)

    def handle(request: httpx.Request) -> httpx.Response:
        peer = "student_a" if request.url.host == "peer-a" else "student_b"
        if peer in down:
            return httpx.Response(503, json={"error": "down"})
        token = __import__("json").loads(request.content)["token"]
        row = feats[feats["token"] == token].iloc[-1]
        owned = ["Return20", "Return5"] if peer == "student_a" else ["MA5", "MA20"]
        return httpx.Response(200, json={"token": token, "as_of_date": str(row["date"].date()),
                                         "features": {k: float(row[k]) for k in owned}})

    return handle


@pytest.fixture
def client(monkeypatch):
    def make(down: set[str] = frozenset(), configured: bool = True, kraken_fail: bool = False):
        peers.clear_cache()
        if configured:
            monkeypatch.setenv("STUDENT_A_URL", "http://peer-a")
            monkeypatch.setenv("STUDENT_B_URL", "http://peer-b")
        else:
            monkeypatch.delenv("STUDENT_A_URL", raising=False)
            monkeypatch.delenv("STUDENT_B_URL", raising=False)
        transport = httpx.MockTransport(peer_handler(down))
        real = httpx.AsyncClient
        monkeypatch.setattr(peers.httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))
        tc = TestClient(main.app)
        tc.__enter__()
        main.app.state.market = FakeKraken(fail=kraken_fail)
        return tc

    return make


# ---------------------------------------------------------------- metadata endpoints
def test_root_lists_endpoints(client):
    body = client().get("/").json()
    assert body["endpoints"] == ["/", "/health", "/features", "/predict", "/model-metadata"]
    assert body["algorithm"] == "XGBoost Classifier" and "github" in body


def test_health(client):
    body = client().get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_model_metadata(client):
    body = client().get("/model-metadata").json()
    assert body["forecast_horizon"] == 7
    assert body["classification_labels"] == TC_LABELS
    assert set(body["feature_importance"]) == set(body["features"])
    assert abs(sum(body["feature_importance"].values()) - 1) < 1e-3


# ---------------------------------------------------------------- /features
def test_features_owned_and_history(client):
    r = client().post("/features", json={"token": "btc", "forecast_horizon": 7, "history_days": 30})
    assert r.status_code == 200
    body = r.json()
    assert body["token"] == "BTC" and len(body["history"]) == 30
    for name in ["Volatility20", "MaxPrice20", "MaxDrawdown", "GapRisk"]:
        assert isinstance(body["features"][name], float)
    assert all("close" in row for row in body["history"])


def test_features_match_contract_formulas(client):
    body = client().post("/features", json={"token": "DOGE"}).json()
    d = FakeKraken().data.query("token == 'DOGE'").reset_index(drop=True)
    ret = d["close"].pct_change()
    assert body["features"]["Volatility20"] == pytest.approx(ret.tail(20).std(ddof=1), rel=1e-6)
    assert body["features"]["MaxPrice20"] == pytest.approx(d["high"].tail(20).max(), rel=1e-9)
    gap = ((d["open"] - d["close"].shift(1)).abs() / d["close"].shift(1)).tail(20).mean()
    assert body["features"]["GapRisk"] == pytest.approx(gap, rel=1e-6)


# ---------------------------------------------------------------- /predict
def test_predict_contract_shape(client):
    r = client().post("/predict", json={"token": "BTC", "forecast_horizon": 7})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in TC_LABELS
    assert 0 < body["confidence"] <= 1
    assert sum(body["probabilities"].values()) == pytest.approx(1, abs=1e-3)
    assert body["confidence"] == max(body["probabilities"].values())
    assert body["forecast_horizon"] == 7
    assert pd.Timestamp(body["prediction_date"]) - pd.Timestamp(body["as_of_date"]) == pd.Timedelta(days=7)
    assert body["missing_features"] == []
    assert body["explanation"][0]["feature"] in body["features_used"]


def test_serving_reproduces_training_path(client):
    """Same day, same data: API probabilities == notebook model probabilities."""
    body = client().post("/predict", json={"token": "DOGE", "forecast_horizon": 7}).json()
    bundle = main.app.state.bundle
    local = compute_local_features(FakeKraken().data.query("token == 'DOGE'"))
    local["MA5_MA20_Ratio"] = local["MA5"] / local["MA20"]
    row = local.iloc[[-1]][bundle["feature_columns"]]
    expected = bundle["model"].predict_proba(row)[0]
    got = np.array([body["probabilities"][label] for label in TC_LABELS])
    np.testing.assert_allclose(got, expected, atol=1e-4)


@pytest.mark.parametrize("down", [{"student_a"}, {"student_b"}, {"student_a", "student_b"}])
def test_predict_survives_peer_outage(client, down):
    body = client(down=down).post("/predict", json={"token": "BTC"}).json()
    assert body["prediction"] in TC_LABELS
    assert "Return20" in body["missing_features"] if "student_a" in down else True
    assert "MA5_MA20_Ratio" in body["missing_features"] if "student_b" in down else True
    assert "TrendScoreProj7" not in body["missing_features"]
    assert all(body["peer_status"][p].startswith("unavailable") for p in down)


def test_predict_without_configured_peers(client):
    r = client(configured=False).post("/predict", json={"token": "BTC"})
    assert r.status_code == 200 and "Return20" in r.json()["missing_features"]


# ---------------------------------------------------------------- errors (CONTRACT 6.6)
def test_unsupported_token(client):
    r = client().post("/predict", json={"token": "ADA", "forecast_horizon": 7})
    assert r.status_code == 400
    assert r.json() == {"error": "Unsupported cryptocurrency",
                        "supported_tokens": ["BTC", "ETH", "XRP", "SOL", "DOGE", "XLM", "TRX", "XMR"]}


def test_unsupported_horizon(client):
    r = client().post("/predict", json={"token": "BTC", "forecast_horizon": 3})
    assert r.status_code == 422 and r.json()["error"] == "Unsupported forecast_horizon"


def test_invalid_request(client):
    r = client().post("/predict", json={"forecast_horizon": 7})
    assert r.status_code == 422 and r.json()["error"] == "Invalid request"


def test_kraken_down(client):
    r = client(kraken_fail=True).post("/predict", json={"token": "BTC"})
    assert r.status_code == 503 and r.json()["error"] == "Market data source unavailable"


def test_tc_boundaries():
    for value, label in {-0.1001: 0, -0.10: 1, -0.02: 2, 0.02: 2, 0.0201: 3, 0.10: 3, 0.1001: 4}.items():
        assert classify_tc(value) == label
