import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold


from cdc_ml.modeling.config import CV_SPLITS, RANDOM_STATE


def make_holdout_split(df: pd.DataFrame):
    sgkf = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    tr, te = next(sgkf.split(df, y=df["has_booking"], groups=df["username"]))
    df_train, df_test = df.iloc[tr], df.iloc[te]
    print(f"Train share -> {len(df_train) / len(df)}")
    print(f"Test share -> {len(df_test) / len(df)}")
    print(f"Baseline positive rate -> {df["has_booking"].mean()}")
    print(f"Train positive rate -> {df_train["has_booking"].mean()}")
    print(f"Test positive rate -> {df_test["has_booking"].mean()}")
    return df_train, df_test
