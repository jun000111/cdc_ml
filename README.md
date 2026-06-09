# Booking Prediction Model (`cdc_ml`)

Predicting whether a single poll will produce a bookable lesson slot, so the polling system can **drop low-value polls without losing bookings**.

This model sits on top of a production polling bot. The bot re-checks each customer's preferred day/timeslot combinations every few minutes, 24/7, and grabs slots that free up when other learners cancel. Polling is cheap individually but expensive in aggregate. The model ranks upcoming polls by booking probability so the system can retain only the highest-value subset.

The companion notebook (`4.01-report`) is the full writeup; this README is the summary.

---

## Problem framing

The booking signal is **supply-side**: a slot only appears when someone else cancels. So the question isn't "does this customer want this slot" (they already declared it as a preference) but "is a matching slot likely to free up around this time." That reframes the whole problem around *when* polls happen and *how broad* a customer's preferences are, rather than customer intent.

The model supports exactly one decision: **rank a customer's upcoming polls by predicted probability and execute only the top fraction.**

---

## Data

| | |
|---|---|
| Source | Real polling logs + customer-type table + preference windows (identifiers anonymized) |
| Grain | One row per `(customer, polling_hour)` — poll timestamps aren't retained, so hourly is the finest resolution |
| Size | 37 customers · 29,612 rows · 390 bookings (**1.3% base rate**) |
| Span | ~9 months (Aug 2025 – Apr 2026) |
| Validation | All cleaned/derived tables enforced with **Pandera** (timestamp integrity, `pref_start < pref_end`, range/sanity checks, binary-column constraints) |

**Two biases that shape what the metrics mean** — and stating them up front is part of the analysis:

1. **Sample is conditioned on success.** Nearly all retained cycles contain ≥1 booking, which over-represents easy-to-convert configurations. It removes few one-team customers (who book easily) but many common-pool customers, so the observed gap between the two groups is *compressed* relative to the true population.
2. **Polling is preference-conditioned.** The bot only polls inside declared preferences, so the data only covers preferred day/timeslot regions. Predictions outside them are extrapolation — but this matches deployment exactly, since the live system also only scores preference-conditioned polls.

Reported metrics therefore describe **relative ranking quality within this sample**, not population-level booking forecasts.

---

## Evaluation design

- **Group split by `username`** (both holdout and CV). Polls from one customer share near-identical timing/preference signatures, so a row-level split would leak. `username` is a grouping key only, never a feature. The objective is **cold-start generalization to unseen customers.**
- **`StratifiedGroupKFold`** preserves customer separation while holding the ~1.3% positive rate roughly constant across folds. ~20% of customers held out as test (train: 30 users / test: 7 users).
- **PR-AUC is the primary metric.** At this prevalence ROC-AUC stays misleadingly high on the negatives; PR-AUC focuses on the booking class. Brier is a secondary (relative) calibration measure; the gains curve is the operational metric.
- **Uncertainty via customer-level bootstrap, not repeated seeds.** A handful of high-volume "whale" customers dominate the data, so reseeding mostly reshuffles the same whales between folds — it doesn't probe the real source of variance. Resampling *customers* does.
- **Whale vs non-whale reported separately throughout.** ~8 customers drive ~half of polling and ~5 drive ~half of bookings, so pooled metrics can hide segment-level behavior. A feature is only kept if it helps the segment of interest, not just the pooled number.

---

## Features

A funnel: four candidate families filtered by three criteria — **available at inference**, **non-redundant**, and **demonstrated lift**.

| Family | Outcome |
|---|---|
| Polling (`polling_hour`, `polling_dow`, …) | **Kept** the two timing features; dropped `polling_day` (redundant with `dow` by Cramér's V) and `polling_month` (only 9 months, would extrapolate) |
| Preference (`pref_unique_timeslot`, `pref_coverage`, `pref_valid`, …) | Collapsed to **`pref_unique_timeslot`** — the rest were correlated summaries of the same window |
| Customer-type (`class_type`, `is_one_team`) | **Kept both** — major axis of booking difficulty |
| Cycle (`cycle_start_*`, `hours_into_cycle`) | **Dropped entirely** — some unavailable at inference (`cycle_end`), others redundant with polling features, and a `00:00` backlog-default artifact contaminates `cycle_start_hour` |

**Final feature set (5):** `polling_hour`, `polling_dow`, `pref_unique_timeslot`, `class_type`, `is_one_team`.

Preference features are deterministic functions of the cycle config (declared at creation), so they introduce **no fold-specific fitting and no leakage pathway** — distinct from leaky per-fold target aggregates.

---

## Baseline

Smoothed lookup tables on timing only (`polling_dow` + `polling_hour`), shrunk toward the base-rate logit to stabilize sparse cells:

- **Additive LUT vs joint `dow × hour` LUT** are statistically indistinguishable (additive leads by +0.003 PR-AUC against a fold-to-fold SD of 0.013) → **no meaningful weekday×hour interaction.** SHAP later agrees.
- Random Forest and XGBoost on the same features **converge to the additive LUT** — extra flexibility extracts nothing further from timing alone.
- Time-only ceiling ≈ **2.1× base rate.** Any further lift must come from preference/customer features, not model complexity.

---

## Feature ablation

On a small, imbalanced dataset, individual PR-AUC deltas are treated as **effect-size estimates, not significance tests.** Decisions rest on redundancy, parsimony, and effect size. Two passes: leave-one-out (marginal contribution) and group ablation (removing whole correlated blocks, since correlated features mask each other).

| Stage | Action | Effect on pooled PR-AUC |
|---|---|---|
| 1 | Drop `pref_dow` (7 sparse weekday counts) | ≈ none → removed |
| 2 | Collapse preference block to `pref_unique_timeslot` | Removing `coverage`/`unique_day`/`valid` together → no loss |
| 3 | Validate the final 5 | Removing `class_type` −38%, `is_one_team` −43%, `pref_unique_timeslot` −54% |

The instructive part: removing `pref_unique_timeslot` cost **~18% in the 8-feature model but ~54% in the 5-feature model.** The feature didn't get more predictive — pruning correlated preference variables **un-masked** its unique contribution.

---

## Model, tuning & calibration

- **XGBoost.** `RandomizedSearchCV` (`n_iter=50`) over the continuous hyperparameter space, scored on `average_precision`, under the same `StratifiedGroupKFold`-by-`username` protocol.
- **`refit=False`** — search returns only `best_params_` → persisted to JSON → consumed by `train.py`. Hyperparameter selection and final training (with OOF generation + calibration) stay separate and reproducible from on-disk config, not notebook state.
- **Platt scaling on pooled OOF predictions.** OOF scores come from models that never saw the held-out rows, so they're a representative deployment-like sample and avoid the optimism of calibrating on in-fold scores. Platt over isotonic because, with so few positives, isotonic's stepwise fit is unstable; the empirical curves show a smooth monotonic over-prediction that a 2-parameter logistic handles well.

---

## Results

Cold-start OOF is the primary estimate (aggregates all training customers under the grouped protocol); the untouched test set is a lower but less-biased reference.

| | Users | Base rate | PR-AUC | Lift | 95% CI (lift) |
|---|---|---|---|---|---|
| **OOF (cold-start, dev)** | 30 | 1.4% | 0.071 | **5.1×** | [2.2×, 7.4×] |
| Test (held out) | 7 | 1.1% | 0.034 | 3.1× | [1.7×, 5.6×] |
| Production (retrained, all data) | 37 | 1.3% | 0.047 | 3.6× | [1.8×, 5.6×] |

**On the OOF–test gap:** the intervals overlap heavily and the test point estimate sits inside the OOF CI, so there's no clean separation. The test set is tiny and high-variance (7 customers, 67 positives; its whale segment is effectively one customer), and adversarial validation shows moderate train↔test shift (**AUC ≈ 0.81**, though fixed per-customer preference signatures inflate this with only 7 holdout customers). Note that OOF also served as the *selection* metric for ablation and tuning, so it carries some optimism — the lower test number is consistent with that plus sampling noise, not evidence of a generalization failure.

### Operational impact — gains curve

If only the top X% of polls (by predicted probability) are kept, what share of bookings survives?

| | XGBoost (final) | Additive baseline |
|---|---|---|
| Polls to retain **90%** of bookings | ~60% | ~62% |
| Polls to retain 80% | ~43% | ~48% |
| Polls to retain 95% | ~78% | ~77% |
| Bookings within **top 10%** of polls | **~46%** | ~27% |

Two takeaways: (1) most of the deployable value is already in the **timing signal** — ~90% of bookings retained at ~40% fewer polls is reachable with the time-only baseline; (2) XGBoost's edge concentrates at the **selective end** (46% vs 27% of bookings in the top 10%), i.e. it ranks the best opportunities higher. Headline: an expected **30–50% reduction in polling while retaining ~90% of bookings** (conservative end from OOF).

### Per-customer guardrail

The gains curve is an aggregate and skews toward whales, so recall is also checked **per customer** at a 60% polling budget. OOF: whales retain a median **92%** (worst 65%, n=4), non-whales **100%** (worst 0%, n=26). The zeros come from customers with 1–2 bookings, where one missed slot tanks a small denominator — coarse statistics, not systematic ranking failure. This flags potential "starvation" customers before deployment.

---

## Interpretation (SHAP)

Mean-|SHAP| ranking: `polling_hour` > `pref_unique_timeslot` > `polling_dow` > `class_type` > `is_one_team`.

- **`polling_hour`** — negative overnight/evening, positive across midday, consistent with cancellations clustering in daytime hours (the supply-side story).
- **`pref_unique_timeslot`** — threshold-shaped: 6–7 distinct timeslots contribute positively, ≤5 negatively. Broader preferences = more chances to match a freed slot.
- **SHAP vs ablation** look like they disagree (SHAP ranks `polling_hour` first; ablation hits hardest on `pref_unique_timeslot`) but they measure different things: mean-|SHAP| is average contribution to predicted log-odds (`polling_hour` touches every poll), while ablation measures lost **ranking** ability (`pref_unique_timeslot` does more to separate bookings from non-bookings). Complementary, not contradictory.
- The `polling_hour` dependence plot shows little day-of-week separation — confirming the additive-vs-joint LUT finding that timing effects are additive.

---

## Limitations

- **Sample-specific metrics** — success-conditioned, preference-conditioned sample; numbers are relative ranking quality, not population booking rates.
- **Customer-level uncertainty** — most variance comes from the small customer count (7 in test), with moderate train↔test shift; hence customer-level bootstrap and OOF-as-primary.
- **Offline counterfactual** — the gains curve assumes booking behavior is unchanged under reduced polling; only live deployment confirms real savings.
- **Coarse, preference-conditioned scope** — hourly granularity, preference-window only. Mirrors the existing system, so a scope limit rather than a modeling defect.
- **Seasonality** — 9 months can't establish annual effects.

---

## Operational use

Deployment uses a **score quantile** rather than a fixed probability threshold (e.g. keep the top ~50% of polls), recomputed periodically to track score-distribution drift. Because OOF scores are calibrated, expected recall at any cutoff can be estimated directly from probability mass (Σ retained p / Σ all p) and monitored without waiting for outcomes — as long as calibration holds (the 0.81 adversarial AUC is a watch item here). Per-customer recall runs as a guardrail against uneven impact. The projected polling reduction is a **forecast**; a live experiment is required to confirm it.

---

## Stack

Cookiecutter Data Science layout · `pandera` schemas · `typer` CLI · `loguru` · Parquet storage · XGBoost / scikit-learn / SHAP. Pipeline: `clean_records.py` → `features.py` → tuning → `train.py` (OOF + Platt calibration + artifacts).