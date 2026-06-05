BASELINE = ["polling_hour", "polling_dow"]

PREF = {
    "coverage": ["pref_coverage"],
    "countdown": ["countdown"],
    "unique_day": ["pref_unique_day"],
    "unique_timeslot": ["pref_unique_timeslot"],
    "dow": [
        "pref_dow_count_0",
        "pref_dow_count_1",
        "pref_dow_count_2",
        "pref_dow_count_3",
        "pref_dow_count_4",
        "pref_dow_count_5",
        "pref_dow_count_6",
    ],
}

CLASS_FEATURES = ["is_one_team", "class_type"]


def feats(*parts):
    out = []
    for p in parts:
        out += p
    return out


FEATS_ABL_LIST = {
    "baseline": feats(BASELINE),
    "baseline+pref_dow": feats(BASELINE, PREF["dow"]),
    "baseline+pref_unique_day": feats(BASELINE, PREF["unique_day"]),
    "baseline+pref_unique_timeslot": feats(BASELINE, PREF["unique_timeslot"]),
    "baseline+pref_coverage": feats(BASELINE, PREF["coverage"]),
    "baseline+countdown": feats(BASELINE, PREF["countdown"]),
    "baseline+uday+utime": feats(BASELINE, PREF["unique_day"], PREF["unique_timeslot"]),
    "baseline+cov+cd": feats(BASELINE, PREF["coverage"], PREF["countdown"]),
    "baseline+uday+utime+cd+cov+cls": feats(
        BASELINE,
        PREF["unique_day"],
        PREF["unique_timeslot"],
        PREF["countdown"],
        PREF["coverage"],
        CLASS_FEATURES,
    ),
    "baseline+uday+utime+cd+cov+cls+dow": feats(
        BASELINE,
        PREF["unique_day"],
        PREF["unique_timeslot"],
        PREF["countdown"],
        PREF["coverage"],
        PREF["dow"],
        CLASS_FEATURES,
    ),
}
