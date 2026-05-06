import pandas as pd


def add_effective_mins(cycle_start: pd.Series, polling_time: pd.Series) -> pd.Series:

    results = pd.Series(60, index=cycle_start.index)
    starting_poll_mask = cycle_start == polling_time

    results[starting_poll_mask] = 60 - polling_time.loc[starting_poll_mask].dt.minute
    return results

    # df["effective_mins"] = add_effective_mins(df["cycle_start"], df["polling_hour"])
    # df["polling_hour"] = df["polling_hour"].dt.floor("h")
