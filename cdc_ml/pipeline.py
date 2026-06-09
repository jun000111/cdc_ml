import typer

from cdc_ml.datasets.proper_records.proper_records import (
    fetch_from_disk as fetch_proper_records,
    clean_from_disk as clean_proper_records,
)
from cdc_ml.datasets.pseudo_records.pseudo_records import clean_from_disk as clean_psuedo_records
from cdc_ml.datasets.merge.merge_records import merge_on_disk
from cdc_ml.datasets.cycle.cycle import clean_from_disk as clean_cycle
from cdc_ml.datasets.customer_class.customer_class import (
    fetch_from_disk as fetch_class,
    clean_from_disk as clean_class,
)
from cdc_ml.datasets.poll.poll import generate_on_disk as generate_poll
from cdc_ml.datasets.preference.preference import clean_from_disk as clean_pref
from cdc_ml.features.build_features import build_on_disk
from cdc_ml.modeling.train import train_on_disk

app = typer.Typer()


@app.command()
def pipeline():
    fetch_proper_records()
    clean_proper_records()
    clean_psuedo_records()
    merge_on_disk()
    clean_cycle()
    fetch_class()
    clean_class()
    generate_poll()
    clean_pref()
    build_on_disk()
    train_on_disk(dev=True)
    train_on_disk()


if __name__ == "__main__":
    app()
