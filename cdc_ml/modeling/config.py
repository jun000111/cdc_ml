RANDOM_STATE = 5
CV_SPLITS = 5

FULL = [
    "polling_hour",
    "polling_dow",
    "pref_coverage",
    "pref_valid",
    "pref_unique_day",
    "pref_unique_timeslot",
    "class_type",
    "is_one_team",
]

PREF_DOW = [
    "pref_dow_count_0",
    "pref_dow_count_1",
    "pref_dow_count_2",
    "pref_dow_count_3",
    "pref_dow_count_4",
    "pref_dow_count_5",
    "pref_dow_count_6",
]

FEATS_ABL_LIST = {
    "full": FULL + PREF_DOW,
    "minus_pref_dow": FULL,
    # "minus_pref_coverage": [g for g in FULL if g != "pref_coverage"] + PREF_DOW,
    # "minus_countdown": [g for g in FULL if g != "countdown"] + PREF_DOW,
    # "minus_pref_unique_day": [g for g in FULL if g != "pref_unique_day"] + PREF_DOW,
    # "minus_pref_unique_timeslot": [g for g in FULL if g != "pref_unique_timeslot"] + PREF_DOW,
    # "minus_class_type": [g for g in FULL if g != "class_type"] + PREF_DOW,
    "minus_is_one_team": [g for g in FULL if g != "is_one_team"] + PREF_DOW,
}

CHOSEN_FEATURES = "minus_pref_dow"

PARAM_DIST = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "min_child_weight": [1, 5, 10, 20, 50],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [1, 5, 10, 20],
    "gamma": [0, 0.5, 1.0, 2.0],
}

XGB_BASE = dict(
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
