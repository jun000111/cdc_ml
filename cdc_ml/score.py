from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import pandas as pd
from psycopg2.extras import execute_values
from sqlalchemy import create_engine

from cdc_ml.config import DATABASE_URL, EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_FIGURE
from cdc_ml.datasets.cycle.cycle import clean_df as clean_cycle_df
from cdc_ml.datasets.poll.poll import generate_df
from cdc_ml.datasets.proper_records.proper_records import clean_df as clean_records_df
from cdc_ml.modeling.evaluation import production_predictions_to_excel

import typer
from pathlib import Path
from loguru import logger

app = typer.Typer()


def get_active_customers(df):
    return df.loc[df["active"] == 1]


def generate_predictions(active):
    tomorrow = (datetime.now(ZoneInfo("Asia/Singapore")) + timedelta(days=1)).date()
    frames = []
    for username, id in (
        active[["username", "id"]].drop_duplicates().itertuples(index=False, name=None)
    ):
        out = production_predictions_to_excel(
            username, tomorrow.month, tomorrow.day, id, retention=0.7
        )
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def push_predictions_to_db(df, engine):
    rows = list(
        df[["username", "polling_date", "polling_hour", "score"]].itertuples(
            index=False, name=None
        )
    )
    sql = """
        INSERT INTO poll_scores (username, polling_date, polling_hour, score)
        VALUES %s
        ON CONFLICT (username, polling_date, polling_hour) DO NOTHING
    """
    conn = engine.raw_connection()
    try:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows)
        conn.commit()
    finally:
        conn.close()


def get_records(engine):
    df_records = pd.read_sql(
        "select * from records where created_at >= '2026-06-12' and username != 'Jun';", engine
    )
    df_records = clean_records_df(df_records)
    df_records.insert(0, "id", range(len(df_records)))
    df_records = df_records.loc[
        ~df_records["username"].isin(["Jun", "modta 🍼", "aditi!", "mal", "r", "mandy"])
    ]
    return df_records


def get_cycle(df_cycle):
    return clean_cycle_df(df_cycle.assign(preference=None, range=None))


def build_poll_df(df_scores, df_records, df_cycle):
    df_poll = generate_df(df_cycle, df_records)

    at = pd.to_datetime(df_poll["polling_at"])
    p = df_poll.assign(polling_date=at.dt.date, polling_hour=at.dt.hour)
    s = df_scores.assign(polling_date=pd.to_datetime(df_scores["polling_date"]).dt.date)
    df = p.merge(
        s[["username", "polling_date", "polling_hour", "score"]],
        on=["username", "polling_date", "polling_hour"],
    )
    df["y"] = df["has_booking"].astype(int)
    return df


def plot_calibration(df, fig_path):
    df["bin"] = pd.qcut(df["score"], 10, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(predicted=("score", "mean"), actual=("y", "mean"))

    plt.plot(g["predicted"], g["actual"], "o-")
    plt.plot([0, g.max().max()], [0, g.max().max()], "k--")  # perfect line
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.savefig(fig_path / f"poll_scores_{date.today().isoformat()}.png")
    # plt.show()

    print(g)
    return g


@app.command()
def main(
    external_input_pref_path: Path = EXTERNAL_DATA_DIR / "cus_pref.xlsx",
    external_input_cycle_path: Path = EXTERNAL_DATA_DIR / "cus_cycle.xlsx",
    report_out_path: Path = REPORTS_FIGURE,
):
    logger.info("Starting...")
    engine = create_engine(DATABASE_URL)

    pref = pd.read_excel(external_input_pref_path)
    cycle = pd.read_excel(external_input_cycle_path)

    logger.success("Retrieved pref and cycle data")
    active = get_active_customers(pref)
    combined = generate_predictions(active)

    df_scores = combined
    logger.success("Generated df_scores")

    push_predictions_to_db(combined, engine)
    logger.success("Pushed to database")

    df_records = get_records(engine)
    df_cycle = get_cycle(cycle)
    df = build_poll_df(df_scores, df_records, df_cycle)

    plot_calibration(df, report_out_path)
    logger.success("Saved plot")


if __name__ == "__main__":
    app()
