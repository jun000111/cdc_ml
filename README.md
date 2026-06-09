# cdc_ml — Booking Prediction for a Slot-Polling Automation System

A machine-learning layer built on top of a production booking-automation system. The system polls a driving-centre booking portal every 3–5 minutes per customer, 24/7, and claims newly-available slots that match customer preferences. This project adds a ranking model that prioritises which polls to execute, reducing polling volume while preserving booking outcomes.

---

## Results summary

| Metric | Value |
|--------|-------|
| PR-AUC (dev OOF, 5-fold grouped CV) | **0.071 · 5.2× over base rate** |
| 95% bootstrap CI | [2.2×, 7.4×] |
| Polling reduction at ~90% booking retention | **~40%** |
| Bookings captured within top 10% of polls | **~46%** |

> **Evaluation note.** The primary metric is out-of-fold (OOF) PR-AUC under `StratifiedGroupKFold` grouped by customer. Each fold withholds a disjoint set of customers entirely, so the OOF estimates cold-start performance on customers the model has never seen. Base rate is ~1.3% (severe class imbalance), making PR-AUC the appropriate primary metric.

---

## System overview

```
Customer ──► Preference config ──────────────────────┐
                                                      │
Booking portal ──► Slot released ──► Poll captured ──►│──► ML model ──► ranked poll queue ──► bot executes top-N
                                                      │
                                         Historical   │
                                         polling log ─┘
```

The end-to-end system comprises:

- **Polling bot** — checks availability every 3–5 min per customer, books on match
- **Backend API** — manages polling state, customer preferences, and booking records
- **Frontend** — customer-facing preference management
- **Telegram notifications** — real-time booking alerts
- **`cdc_ml`** — ML layer that scores each pending poll and ranks by predicted booking probability

---

## Problem framing

A **poll** is one check of the portal for one customer at one hour. The label `has_booking` is `True` if that poll resulted in a booking. The model estimates P(booking | poll context), where context is the poll timing and the customer's preference window.

The booking signal is **supply-side**: a slot only becomes available when another learner cancels. Booking probability is therefore driven by when cancellations tend to occur and how many preference slots a customer has open (preference features).

Class imbalance is severe: approximately 1.3% of polls result in a booking.

---

## Model

**Architecture:** XGBoost classifier with Platt scaling calibration  
**Features (5 total):**

| Feature | Type | Rationale |
|---------|------|-----------|
| `polling_hour` | Temporal | Strongest predictor; cancellations cluster during daytime |
| `polling_dow` | Temporal | Weekend penalty observed in booking rates |
| `pref_unique_timeslot` | Preference | Threshold effect at 6–7 distinct timeslots; wider windows are easier to satisfy |
| `class_type` | Customer | Separates booking difficulty by pool type |
| `is_one_team` | Customer | One-team customers have wider slot access |

**Calibration:** Platt scaling on pooled OOF predictions. Isotonic regression ruled out — too few positives (~1.3%) for a stable non-parametric fit.

---

## Evaluation design

- **Split:** `StratifiedGroupKFold` grouped by `username` — each customer appears entirely in one partition
- **Goal:** cold-start generalisation to customers not seen during training
- **Primary metric:** PR-AUC (average precision) — appropriate under severe imbalance; ROC-AUC dominated by the large negative class
- **Uncertainty:** customer-level bootstrap on OOF predictions (resamples customers, not rows)
- **Baselines:** additive look-up table (DOW + hour), joint DOW×hour table, marginal tables, Random Forest

---

## Operational impact (OOF, 30 training customers)

The gains curve shows how many bookings are retained as the polling budget decreases:

| Polling budget | Bookings retained | Model | Additive baseline |
|---------------|------------------|-------|-------------------|
| Top 10% of polls | ~46% | ← | ~27% |
| Top 21% of polls | ~60% | ← | — |
| Top 28% of polls | ~70% | ← | — |
| Top 42% of polls | ~80% | ← | ~48% |
| Top 60% of polls | ~90% | ← | ~62% |

The largest advantage over the timing-only baseline appears at the most selective end of the ranking. At 90% booking retention, the gap is small (~2 percentage points). The model adds most value when polling must be cut to below 30% of its original volume.

### Per-customer guardrail (60% budget)

Aggregate gains curves can mask uneven distribution across customers.

| Segment | Median recall | Min recall | n |
|---------|-------------|-----------|---|
| Whales (OOF) | 92% | 65% | 4 |
| Non-whales (OOF) | 100% | 0% | 26 |
| Whales (test) | 98% | 95% | 2 |
| Non-whales (test) | 100% | 100% | 5 |

The 0% minimum for a non-whale customer reflects a low-volume case (1–2 bookings) where a single booking fell below the cutoff — a denominator effect rather than a systematic failure.

---

## Repository structure

```
cdc_ml/
├── cdc_ml/
│   ├── config.py                  # paths and constants
│   ├── datasets/
│   │   ├── preference/
│   │   │   └── schema.py          # pandera schemas (RawRecords, CleanedRecords, CleanedPreference)
│   │   └── cycle/
│   ├── features/
│   │   └── build_features.py      # feature engineering pipeline
│   ├── modeling/
│   │   ├── data.py                # train/test split (make_holdout_split)
│   │   ├── train.py               # training pipeline (CLI via typer)
│   │   └── predict.py             # inference
│   └── plots.py                   # all visualisation helpers
├── models/
│   ├── booking_model_v1.joblib    # production model (XGBoost + Platt)
│   └── best_params.json           # hyperparameters from RandomizedSearchCV
├── data/
│   ├── processed/
│   │   ├── polls_processed.parquet
│   │   ├── customer_class_processed.parquet
│   │   └── preference_processed.parquet
│   └── metrics/
│       ├── dev_oof_metrics.parquet
│       └── prod_oof_metrics.parquet
├── reports/
│   └── figures/
├── notebooks/
│   └── 4_01-report.html           # this notebook
└── pyproject.toml
```

---

## Reproducing the model

```bash
# 1. Install
pip install -e .

# 2. Train (reads best_params.json, writes booking_model_v1.joblib + dev_oof_metrics.parquet)
python -m cdc_ml.modeling.train

# 3. Predict for a customer
python -m cdc_ml.modeling.predict --username <name>
```

Data cleaning and feature engineering run automatically as part of the pipeline. Pandera schemas enforce type and constraint validation on all intermediate tables.

---

## Known limitations

1. **Sample conditioned on success.** Only cycles with ≥1 booking are retained. The dataset therefore describes a more easily-booked subset of the full customer base; reported booking rates are not population-level estimates.

2. **No unseen holdout for the production model.** The production model is trained on all 37 customers. The only honest cold-start estimate for genuinely new customers is the dev test result (3.1× lift, 7 held-out users), which is noisy at that panel size.

3. **Strong train–test distribution shift.** Adversarial validation yields AUC = 0.807, indicating the held-out test customers are substantially different from the training population. Preference features, which are per-user constants, are the primary driver.

4. **Offline counterfactual evaluation.** Gains-curve results assume booking and cancellation behaviour would be unchanged under reduced polling. Live deployment is required to confirm operational savings.

5. **Coarse time resolution.** Polls are aggregated to one-hour bins. Within-hour timing is unavailable.

6. **No annual seasonality estimate.** The dataset spans ~9 months, which is insufficient to separate week-over-week patterns from any longer seasonal trend.

---

## Background

The polling bot and its supporting infrastructure (backend, frontend, database, Telegram notifications) were built from scratch as a separate project. `cdc_ml` is an ML layer added on top of that production system. The bot was operational before any ML work began; the model is therefore a performance optimisation, not a core dependency.