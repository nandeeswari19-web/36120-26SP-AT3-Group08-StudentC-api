# 36120-26SP-AT3-Group08 – Student C API: Trend Category (TC)

**FastAPI service forecasting the Trend Category of a cryptocurrency 7 days ahead with a tuned XGBoost Classifier, and publishing Student C's engineered features for the other Group 08 services.**

| | |
|---|---|
| **Subject** | 36120 Advanced Machine Learning Application, Spring 2026, UTS |
| **Assessment** | AT3 – Data Product with Machine Learning |
| **Group** | 08 |
| **Student** | Student C – Nandeeswari Kumaravelu – 25980436 |
| **Deployment** | `https://<student-c-service>.onrender.com` _(add after deploying)_ |
| **Repository** | `https://github.com/<github-username>/36120-26SP-AT3-Group08-StudentC-api` |
| **Experiments** | `notebooks/student_c/` and `at3_crypto/modeling/student_c_tc.py` in `36120-26SP-AT3-Group08-experiments` |
| **Contract** | `CONTRACT.md` in the experiments repository (feature names, request/response shapes, errors) |

### Group members

| Role | Target | Name | Student ID |
|---|---|---|---|
| Student A | Asset Performance Index | Ryan Cruikshank | 13191263 |
| Student B | Market Risk Index | Vignesh Selvam | 25740287 |
| **Student C** | **Trend Category** | **Nandeeswari Kumaravelu** | **25980436** |
| Student D | Trading Opportunity Rating | Vinnie Bui | 26160438 |

---

## 1. What the service does

- **Target:** `TrendScore = 0.7 × ((MA5 / MA20) − 1) + 0.3 × Return20`, measured at **t + 7**, classified as  
  `Strong Downtrend` (< −0.10) · `Downtrend` (−0.10 to < −0.02) · `Sideways` (−0.02 to 0.02, inclusive) · `Uptrend` (> 0.02 to 0.10) · `Strong Uptrend` (> 0.10)
- **Model:** XGBoost Classifier (`multi:softprob`), tuned with Optuna (2 × 50 TPE trials, `TimeSeriesSplit(5, gap=7)` over dates), trained on all 8 tokens.
- **Model inputs:** `Return20`, `Return5` (from Student A), `MA5_MA20_Ratio`, `TrendScore` (from Student B's MA5/MA20 + A's Return20) and **`TrendScoreProj7`**: Student C's flat-price projection of the t+7 TrendScore, computed locally from Kraken closes.
- **Published features (POST /features):** `Volatility20`, `MaxPrice20`, `MaxDrawdown`, `GapRisk` (+ `TrendScoreProj7`).
- **Data:** live Kraken public OHLC API (daily USD candles; the incomplete current day is dropped). Cached for 15 minutes.

### Test performance (held-out 12 Jul – 15 Oct 2026, 768 token-days, scored once)

| Model | macro-F1 | balanced acc. | accuracy | within one class |
|---|---|---|---|---|
| **Tuned XGBoost (deployed)** | **0.4776** | 0.4621 | 0.5417 | 0.8932 |
| Flat-price projection rule | 0.4679 | 0.4629 | 0.5911 | 0.9167 |
| Persistence (today's trend) | 0.4675 | 0.4462 | 0.5091 | 0.8307 |
| XGBoost, default parameters | 0.4284 | 0.4262 | 0.4909 | 0.8672 |
| Majority class | 0.0718 | 0.2000 | 0.2188 | 0.5430 |

Log-loss 1.123, **ECE 0.049** (the confidence score is well calibrated).

**Resilience:** test macro-F1 is 0.478 with all peers up, 0.435 with Student A down, 0.484 with Student B down, and 0.484 with both down. The service always answers, because `TrendScoreProj7` does not depend on any peer.

---

## 2. Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Project metadata, task description, endpoint list, GitHub URL |
| GET | `/health` | `{"status": "ok", "model_loaded": true, "model_version": "1.0.0", "timestamp": ...}` |
| POST | `/features` | Student C's features for a token (+ `history` for charts) |
| POST | `/predict` | 7-day-ahead Trend Category with confidence and probabilities |
| GET | `/model-metadata` | Target, algorithm, horizon, features, feature importance, labels, metrics |
| GET | `/docs` | Interactive Swagger UI |

### POST /predict

```json
{"token": "BTC", "forecast_horizon": 7}
```

```json
{
  "prediction": "Sideways",
  "confidence": 0.4057,
  "probabilities": {"Strong Downtrend": 0.0422, "Downtrend": 0.2811, "Sideways": 0.4057, "Uptrend": 0.2476, "Strong Uptrend": 0.0234},
  "forecast_horizon": 7,
  "prediction_date": "2026-10-29",
  "token": "BTC",
  "as_of_date": "2026-10-22",
  "model_version": "1.0.0",
  "missing_features": [],
  "feature_sources": {"Return20": "student_a_api", "MA5": "student_b_api", "TrendScoreProj7": "student_c (local, Kraken)"},
  "peer_status": {"student_a": "ok", "student_b": "ok"},
  "features_used": {"Return20": -0.0377, "MA5_MA20_Ratio": 0.9929, "TrendScore": -0.0163, "Return5": -0.0008, "TrendScoreProj7": -0.0101},
  "explanation": [{"feature": "TrendScoreProj7", "value": -0.0101, "contribution": 0.1857}],
  "current": {"date": "2026-10-22", "close": 81315.9, "trend": "Sideways", "trend_score": -0.0163, "MA5": 82423.98, "MA20": 83010.14}
}
```

`explanation` contains SHAP contributions (XGBoost `pred_contribs`, log-odds) for the predicted class. `current` describes today's trend from the same formulas, for display.

### POST /features

```json
{"token": "BTC", "forecast_horizon": 7, "history_days": 1}
```
Returns `token`, `as_of_date`, `source`, `features` {`Volatility20`, `MaxPrice20`, `MaxDrawdown`, `GapRisk`, `TrendScoreProj7`}, `definitions` and `history` (last `history_days` rows, 1–365, oldest first, always with `close`). Values that cannot be computed are `null`.

### Errors (same shape everywhere)

| Situation | HTTP | Body |
|---|---|---|
| Unknown token | 400 | `{"error": "Unsupported cryptocurrency", "supported_tokens": ["BTC","ETH","XRP","SOL","DOGE","XLM","TRX","XMR"]}` |
| `forecast_horizon` ≠ 7 | 422 | `{"error": "Unsupported forecast_horizon", "supported_horizons": [7]}` |
| Malformed body | 422 | `{"error": "Invalid request", "details": [...]}` |
| Kraken unavailable | 503 | `{"error": "Market data source unavailable", "retry_after_seconds": 60}` |
| Unexpected | 500 | `{"error": "Internal server error"}` |

---

## 3. How a prediction is made

1. Download daily candles for the token from Kraken (drop today's incomplete candle; cache 15 min).
2. Compute Student C's features and `TrendScoreProj7` locally with the shared group code (`app/core/`).
3. Call **Student A** (`Return20`, `Return5`) and **Student B** (`MA5`, `MA20`) `POST /features` **in parallel** (httpx, 10 s timeout, 1 retry, 15-min cache).
4. If a peer is not configured, asleep or failing, its features become **NaN** and are listed in `missing_features`. XGBoost routes NaN through branches learned during training (the model was trained with missing-feature augmentation), so the request still returns **200**.
5. Predict class probabilities, then return the most likely class, its probability (`confidence`), and SHAP explanations.

---

## 4. Setup and run locally

```bash
git clone https://github.com/<github-username>/36120-26SP-AT3-Group08-StudentC-api.git
cd 36120-26SP-AT3-Group08-StudentC-api
python3.12 -m venv .venv && source .venv/bin/activate      # Python 3.12.13
pip install -r requirements-dev.txt
cp .env.example .env    # optional: set STUDENT_A_URL / STUDENT_B_URL
uvicorn app.main:app --reload
# open http://127.0.0.1:8000/docs
```

Run the tests (offline: Kraken and the peers are mocked with the committed snapshot):

```bash
pytest -q
```

The tests cover:
- every endpoint and error code
- the contract formulas of the published features
- a peer outage (A, B, both, or not configured)
- the boundaries of the trend classes
- **serving = training**: the API's probabilities equal the notebook model's for the same day

---

## 5. Deploy on Render

1. Push this repository to GitHub (private). Give the teaching staff administrator access.
2. Render → **New → Web Service** → connect the repository. Or use **New → Blueprint**, which reads `render.yaml`.
3. **Build command:** `pip install -r requirements.txt`  
   **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. **Environment variables:**
   - `PYTHON_VERSION=3.12.13`
   - `STUDENT_A_URL=<Student A Render URL>`
   - `STUDENT_B_URL=https://group08-studentb-mri-api.onrender.com`
   - `GITHUB_REPO_URL=<this repo URL>`
5. Health check path: `/health`. After deploying, open `/docs` and run `/predict` for `BTC`.

Free Render instances sleep after inactivity; the first call can take about 50 seconds. Call `/health` to wake the service before a demo.

---

## 6. Repository structure

```
├── app/
│   ├── main.py              <- FastAPI app: endpoints, validation, error handling
│   ├── features.py          <- Student C feature assembly (same formulas as training)
│   ├── peers.py             <- Parallel calls to Student A / B /features with timeout, retry, cache
│   └── core/                <- Verbatim copies of the group modules (at3_crypto/):
│       ├── kraken.py        <-   Kraken OHLC client (CONTRACT.md §3)
│       ├── engineered_features.py <- all CONTRACT.md §4 features
│       └── targets.py       <-   TrendScore / TC classification (CONTRACT.md §5)
├── models/
│   ├── tc_xgboost.joblib    <- fitted model bundle (model, feature list, labels, metadata)
│   ├── tc_xgboost.json      <- same model in XGBoost's native format
│   └── metrics.json         <- metadata served by /model-metadata (written by the notebook)
├── tests/                   <- offline pytest suite + Kraken snapshot fixture
├── requirements.txt         <- pinned runtime dependencies
├── pyproject.toml
├── render.yaml / Dockerfile <- deployment
└── README.md
```

## 7. AI usage acknowledgement

Generative AI (Claude) was used to support coding and documentation, in line with the subject's AI policy. All content was reviewed, executed and validated by the student, who takes responsibility for it and can explain every design decision.
