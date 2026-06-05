import json, datetime
from pathlib import Path
from cdc_ml.config import REPORTS_DIR

RESULTS = Path(REPORTS_DIR / "results.jsonl")


def log_run(config: dict, metrics: dict):
    record = {
        "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
        **config,  # what you ran:  {"model": "xgboost", "ablation": "pref_dow_only"}
        **metrics,  # what you got:  {"pr_auc_oof": 0.34, "pr_auc_ci": [0.30, 0.38]}
    }
    with RESULTS.open("a") as f:
        f.write(json.dumps(record) + "\n")
