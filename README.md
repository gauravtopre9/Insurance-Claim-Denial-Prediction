# Insurance Claim Denial Prediction — Pipeline

An end-to-end machine learning pipeline that predicts which insurance claims
are likely to be denied, so review teams can prioritize the highest-risk
claims before submission. The pipeline trains multiple model families,
selects the best performer using a capacity-constrained recall metric,
explains its predictions with SHAP, and generates plain-English,
LLM-written risk explanations for the highest- and lowest-risk claims.

## Setup

### 1. Requirements

```bash
pip install -r requirements.txt
```

### 2. Data

Place these two files in the project root (same folder as `main.py`):

- `claims_history.csv` — labeled historical claims with a `split` column
  (`train` / `validation` / `test`) and an `is_denied` target
- `current_claims.csv` — unlabeled claims to be scored

### 3. Configuration

Review `config.yaml` before running:

- `col_details.CAT_COLS` / `CUST_COL` — categorical columns to encode
- `col_details.DROP` — columns excluded from the model's feature set
- `params.RS` — random seed
- `params.K` — the review capacity fraction (e.g. `0.25` = top 25% by
  predicted risk gets reviewed)
- `params.N_FOLDS` / `N_TRIALS` — cross-validation folds and Optuna trials
  used during hyperparameter tuning

**Note:** `config.yaml` in this deliverable is a reconstructed best-guess
(it wasn't part of the original source material), so please confirm the
column lists match your actual dataset before relying on it.

### 4. LLM explanations (optional stage)

The final stage uses Groq (via LangChain) to write natural-language
explanations for the top and bottom 10 scored claims. Set a valid API key
before running that stage in .env file:

```bash
GROQ_API_KEY=MY_GROQ_API_KEY
```

or pass it directly: `setup_llm(groq_api_key="MY_GROQ_API_KEY")` in
`src/llm_insights.py`. If you don't need this stage, you can skip calling
it in `main.py` and everything else in the pipeline runs independently.

### 5. Run

```bash
python main.py
```

This produces:
- `Predictions_current_claims.csv` — every current claim scored with a
  denial probability and a top-25%-review flag
- `top10_claim_explanations.csv` — LLM explanations for the highest-risk claims
- `bottom10_claim_explanations.csv` — LLM explanations for the lowest-risk claims

## Approach

**Problem framing.** Review teams can't manually check every claim before
submission — they can only review a fixed percentage of the volume. So
instead of optimizing for generic classification accuracy, the model is
tuned and evaluated on **Recall@K**: of all the claims that will actually be
denied, what fraction get caught if the team reviews only the top K% by
predicted risk? This is the metric used for every model comparison,
hyperparameter search, and final model selection decision.

**Feature engineering.** Raw claim fields are turned into signal along a
few themes:
- *Administrative/compliance gaps* — missing prior authorization, missing
  referrals, unverified eligibility, out-of-network status, and combinations
  of these ("compliance failure count").
- *Financial ratios* — billed-vs-expected ratio, cost per procedure/
  diagnosis, high-cost flags.
- *Timeliness* — days-to-submit thresholds and log-transformed submission
  delay.
- *Payer behavior* — payer-level historical denial rates, average billed/
  expected amounts, and payer × visit-type / payer × auth-gap importance
  scores, all computed strictly from the training split to avoid leaking
  future information into features.
- *Compound interactions* — combinations of the above (e.g. high-cost AND
  out-of-network) that tend to compound denial risk.

**Encoding.** Categorical fields are both label-encoded (for tree-based
models) and target-encoded with smoothing (shrinkage toward the global mean
for low-frequency categories), fit only on the training split and applied
consistently everywhere else.

**Modeling.** Four model families — Logistic Regression, Random Forest,
LightGBM, and XGBoost — are each trained three ways:
1. a sensible baseline configuration,
2. Optuna-tuned hyperparameters (optimizing cross-validated Recall@K), and
3. trained on SMOTE-oversampled data to test whether synthetic minority
   oversampling helps.

For each model family, whichever of the three variants scores highest on
the validation set is kept. The single best model overall is then chosen by
running each finalist against the test set and comparing actual denial
capture rate at the review-capacity threshold — the same metric the
business cares about.

**Error analysis & explainability.** Missed vs. captured denials are broken
down by denial reason, payer type, and visit type to surface where the
model underperforms. SHAP is used to explain which features drive the
model's predictions, both in aggregate (validation set) and per-claim
(current claims being scored).

**Scoring & natural-language insights.** The current, unlabeled claims are
scored and ranked by predicted denial probability. For the highest-risk and
lowest-risk claims, an LLM (Groq/Llama via LangChain) is given the claim's
fields, its model-derived risk drivers (e.g. "referral required but not
present"), and a fixed prompt template — it returns a short, three-sentence,
analyst-readable explanation with a recommended action, so a human reviewer
doesn't have to interpret raw model scores on their own.

**Risk Tiers**

| Risk Tier   | Score      |
|-------------|------------|
| High Risk   | ≥ 75%      |
| Medium Risk | 50% – <75% |
| Low Risk    | < 50%      |