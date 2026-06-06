import json, datetime
import numpy as np
from pathlib import Path
from cdc_ml.config import FEATURE_ABLATIONS_RESULTS


def _to_jsonable(o):
    if isinstance(o, np.generic):  # np.float64, np.int64, ...
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o)} not JSON serializable")


def log_run(config: dict, metrics: dict):
    FEATURE_ABLATIONS_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    record = {"run_at": datetime.datetime.now().isoformat(timespec="seconds"), **config, **metrics}
    with FEATURE_ABLATIONS_RESULTS.open("a") as f:
        f.write(json.dumps(record, default=_to_jsonable) + "\n")
