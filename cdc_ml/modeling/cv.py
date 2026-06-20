from pathlib import Path
import datetime

from loguru import logger
from tqdm import tqdm
import pandas as pd
import typer
import numpy as np
import xgboost as xgb
from sklearn.metrics import brier_score_loss, average_precision_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from cdc_ml.config import MODELS_DIR, STAGE_1_PROCESSED, OOF_ADD
from cdc_ml.modeling.baseline import *
from cdc_ml.modeling.config import CV_SPLITS, RANDOM_STATE
from cdc_ml.features.build_features import get_whale_users
from reports.tracking import log_run

app = typer.Typer()


from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def run_cv(
    df: pd.DataFrame,
    test_set: str,
    feats_list: dict,
    lr=0.03,
    m_depth=2,
    min_child=10,
    reg_lamb=10,
    seed=RANDOM_STATE,
):

    run_at = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    for feature_name, features in feats_list.items():

        y = df["has_booking"].to_numpy()
        X = df[features]
        n = len(y)
        base = y.mean()

        # name -> predictor(X_tr, y_tr, X_va) -> p_va ; insertion order drives report order
        BASELINES = {
            "const": lambda Xt, yt, Xv: baseline_const(Xt, yt),
            "marg_dow": marg_dow_baseline,
            "marg_hour": marg_hour_baseline,
            "add": additive_lut_logit,
            "joint": joint_lut_hier,
        }
        model_names = (*BASELINES, "xgb", "rf")

        oof = {name: np.full(n, np.nan) for name in model_names}
        fold_brier = {name: [] for name in model_names}
        fold_pr = {name: [] for name in model_names}
        tr_brier_list, tr_pr_list = [], []
        models, importances, importances_std = [], [], []

        whales_pc, whales_pt = get_whale_users()
        whales_pt_mask = df["username"].isin(whales_pt).to_numpy()
        whales_pc_mask = df["username"].isin(whales_pc).to_numpy()

        sgkf = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(sgkf.split(X, y=df["has_booking"], groups=df["username"])):
            X_tr, y_tr = X.iloc[tr], y[tr]
            X_va, y_va = X.iloc[va], y[va]

            # --- XGBoost ---
            model = xgb.XGBClassifier(
                n_estimators=500,
                objective="binary:logistic",
                learning_rate=lr,
                max_depth=m_depth,
                min_child_weight=min_child,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=reg_lamb,
                tree_method="hist",
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            models.append(model)

            # --- RandomForest ---
            rf = RandomForestClassifier(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=50,
                n_jobs=-1,
                random_state=seed,
            )
            rf.fit(X_tr, y_tr)
            oof["rf"][va] = rf.predict_proba(X_va)[:, 1]

            r = permutation_importance(
                model,
                X_va,
                y_va,
                n_repeats=10,
                random_state=seed,
                n_jobs=1,
                scoring="average_precision",
            )
            importances.append(r.importances_mean)
            importances_std.append(r.importances_std)

            oof["xgb"][va] = model.predict_proba(X_va)[:, 1]
            tr_pred = model.predict_proba(X_tr)[:, 1]

            # --- baselines ---
            for name, fn in BASELINES.items():
                oof[name][va] = fn(X_tr, y_tr, X_va)

            # --- per-fold scores ---
            fb, fp = {}, {}
            for name in model_names:
                fb[name] = brier_score_loss(y_va, oof[name][va])
                fp[name] = average_precision_score(y_va, oof[name][va])
                fold_brier[name].append(fb[name])
                fold_pr[name].append(fp[name])
            tr_brier = brier_score_loss(y_tr, tr_pred)
            tr_pr = average_precision_score(y_tr, tr_pred)
            tr_brier_list.append(tr_brier)
            tr_pr_list.append(tr_pr)

            # --- fold report ---
            tr_dates, va_dates = df.iloc[tr]["cycle_start"], df.iloc[va]["cycle_start"]
            print(
                f"fold {fold}: "
                # f"train n={len(tr):>6} ({tr_dates.min().date()} → {tr_dates.max().date()})  "
                # f"val n={len(va):>6} ({va_dates.min().date()} → {va_dates.max().date()})\n"
                f"train n={len(tr):>6} "
                f"val n={len(va):>6} "
                f"  train_pos={y_tr.mean():.3f}  val_pos={y_va.mean():.3f}\n"
                f"  const  brier={fb['const']:.4f}  pr={fp['const']:.4f}\n"
                f"  marg_dow  brier={fb['marg_dow']:.4f}  pr={fp['marg_dow']:.4f}\n"
                f"  marg_hour brier={fb['marg_hour']:.4f}  pr={fp['marg_hour']:.4f}\n"
                f"  add       brier={fb['add']:.4f}  pr={fp['add']:.4f}\n"
                f"  joint     brier={fb['joint']:.4f}  pr={fp['joint']:.4f}\n"
                f"  rf        brier={fb['rf']:.4f}  pr={fp['rf']:.4f}\n"
                f"  xgb (val) brier={fb['xgb']:.4f}  pr={fp['xgb']:.4f}\n"
                f"  xgb (tr)  brier={tr_brier:.4f}  pr={tr_pr:.4f}\n"
            )

        # --- segment base rates ---
        whale_pt_base = y[whales_pt_mask].mean()
        non_whale_pt_base = y[~whales_pt_mask].mean()
        whale_pc_base = y[whales_pc_mask].mean()
        non_whale_pc_base = y[~whales_pc_mask].mean()

        def _metrics(name):
            b = brier_score_loss(y, oof[name])
            p = average_precision_score(y, oof[name])
            return {
                "brier_pooled": b,
                "brier_lift": b / (base * (1 - base)),
                "brier_folds": fold_brier[name],
                "pr_auc_pooled": p,
                "pr_auc_lift": p / base,
                "pr_auc_folds": fold_pr[name],
            }

        def _segment(arr):
            wb = brier_score_loss(y[whales_pc_mask], arr[whales_pc_mask])
            nwb = brier_score_loss(y[~whales_pc_mask], arr[~whales_pc_mask])
            wp = average_precision_score(y[whales_pt_mask], arr[whales_pt_mask])
            nwp = average_precision_score(y[~whales_pt_mask], arr[~whales_pt_mask])
            return {
                "whale_brier": wb,
                "whale_brier_lift": wb / whale_pc_base,
                "non_whale_brier": nwb,
                "non_whale_brier_lift": nwb / non_whale_pc_base,
                "whale_pr": wp,
                "whale_pr_lift": wp / whale_pt_base,
                "non_whale_pr": nwp,
                "non_whale_pr_lift": nwp / non_whale_pt_base,
            }

        def _fmt_row(name, m):
            bl, pl = m["brier_folds"], m["pr_auc_folds"]
            return (
                f"{name:<9} brier={m['brier_pooled']:.4f} ({np.mean(bl):.4f}±{np.std(bl):.4f}) lift={m['brier_lift']:.3f}   "
                f"pr_auc={m['pr_auc_pooled']:.4f} ({np.mean(pl):.4f}±{np.std(pl):.4f}) lift={m['pr_auc_lift']:.3f}"
            )

        def _fmt_segment(s):
            return (
                f"          whales     brier={s['whale_brier']:.4f} lift={s['whale_brier_lift']:.3f}\n"
                f"          non-whales brier={s['non_whale_brier']:.4f} lift={s['non_whale_brier_lift']:.3f}\n"
                f"          whales     pr={s['whale_pr']:.4f} lift={s['whale_pr_lift']:.3f}\n"
                f"          non-whales pr={s['non_whale_pr']:.4f} lift={s['non_whale_pr_lift']:.3f}"
            )

        # importance table
        avg_df = pd.DataFrame(
            {
                "feature": X.columns,
                "mean": np.mean(importances, axis=0),
                "std": np.mean(importances_std, axis=0),
            }
        ).sort_values("mean", ascending=False)

        MODELS = ["const", "marg_dow", "marg_hour", "add", "joint", "rf", "xgb"]

        mets = {nm: _metrics(nm) for nm in MODELS}
        seg = {nm: _segment(oof[nm]) for nm in MODELS}

        # ---- print (console only) ----
        lines = [
            f"\nOOF  base_rate={base:.4f}  whale_pr_base_rate={whale_pt_base:.4f}  non_whale_pr_base_rate={non_whale_pt_base:.4f}",
            f"n={len(y)}/{n}      format: pooled (mean±std over folds)",
        ]
        for nm in MODELS:
            lines.append(_fmt_row(nm, mets[nm]))
            if nm in MODELS:
                lines.append(_fmt_segment(seg[nm]))
        lines.append(
            f"xgb_tr    brier=  ---   ({np.mean(tr_brier_list):.4f}±{np.std(tr_brier_list):.4f})   "
            f"pr_auc=  ---   ({np.mean(tr_pr_list):.4f}±{np.std(tr_pr_list):.4f})"
        )
        lines.append(f"Average importances\n{avg_df}")
        print("\n".join(lines))

        # ---- log (structured, one row per model) ----
        shared = {
            "run_at": run_at,
            "run_id": test_set,
            "feature_id": feature_name,
            "split": "stratified_group_kfold",
            "group": "username",
            "n_folds": 5,
            "seed": seed,
            "features": features,
            "n": n,
            "base_rate": base,
        }
        for nm in MODELS:
            cfg = {**shared, "model": nm}
            metrics = dict(mets[nm])
            metrics["whale_pr"] = seg[nm]["whale_pr"]
            metrics["non_whale_pr"] = seg[nm]["non_whale_pr"]
            metrics["whale_pr_lift"] = seg[nm]["whale_pr_lift"]
            metrics["non_whale_pr_lift"] = seg[nm]["non_whale_pr_lift"]
            if nm == "xgb":
                cfg.update(
                    {
                        "learning_rate": lr,
                        "max_depth": m_depth,
                        "min_child": min_child,
                        "reg_lambda": reg_lamb,
                    }
                )
                metrics["pr_auc_train_mean"] = float(
                    np.mean(tr_pr_list)
                )  # memorization fingerprint
                metrics["pr_auc_folds_mean"] = float(np.mean(fold_pr["xgb"]))
                metrics["pr_auc_folds_std"] = float(np.std(fold_pr["xgb"]))
                metrics.update(
                    {f"imp_{r['feature']}": round(r["mean"], 4) for _, r in avg_df.iterrows()}
                )
            if nm == "const":
                metrics["pr_auc_folds_mean"] = float(np.mean(fold_pr["const"]))
                metrics["pr_auc_folds_std"] = float(np.std(fold_pr["const"]))
            if nm == "add":
                metrics["pr_auc_folds_mean"] = float(np.mean(fold_pr["add"]))
                metrics["pr_auc_folds_std"] = float(np.std(fold_pr["add"]))
            if nm == "joint":
                metrics["pr_auc_folds_mean"] = float(np.mean(fold_pr["joint"]))
                metrics["pr_auc_folds_std"] = float(np.std(fold_pr["joint"]))
            if nm == "marg_dow":
                metrics["pr_auc_folds_mean"] = float(np.mean(fold_pr["marg_dow"]))
                metrics["pr_auc_folds_std"] = float(np.std(fold_pr["marg_dow"]))
            if nm == "marg_hour":
                metrics["pr_auc_folds_mean"] = float(np.mean(fold_pr["marg_hour"]))
                metrics["pr_auc_folds_std"] = float(np.std(fold_pr["marg_hour"]))

            log_run(config=cfg, metrics=metrics)

        np.save(OOF_ADD, oof["add"])


def train_on_disk(data: Path, models: Path):
    logger.info("Training some model...")
    df = pd.read_parquet(data)
    sgkf = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    tr, te = next(sgkf.split(df, y=df["has_booking"], groups=df["username"]))
    df_train, df_test = df.iloc[tr], df.iloc[te]

    train(df_train, [])
    logger.success("Modeling training complete.")


@app.command()
def run(
    data_path: Path = STAGE_1_PROCESSED,
    model_path: Path = MODELS_DIR / "model.pkl",
):
    train_on_disk(data_path, model_path)


if __name__ == "__main__":
    app()
