"""Template for the paper's JPM intraday log-volume experiment.

The historical notebook expects a CSV with a `log_volume` column.
Bloomberg-derived data are not redistributed in this repository.
"""

from pathlib import Path

from aggregate_dynamics.data import load_intraday_log_volume


CSV_PATH = Path("data/refined_JPM.csv")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            "Place an authorized local refined_JPM.csv under data/."
        )

    snapshots = load_intraday_log_volume(
        CSV_PATH,
        observations_per_day=78,
        column="log_volume",
    )

    print("snapshot matrix:", snapshots.shape)
    print("rows = intraday 5-minute bins; columns = trading days")

    # Paper-reported training bins correspond to x0, x2, x7, x10, x22.
    training_bins = [0, 2, 7, 10, 22]
    prediction_bins = [1, 9, 13, 21]

    for i in training_bins:
        print("training aggregate", i, snapshots[i].shape)

    for i in prediction_bins:
        print("prediction aggregate", i, snapshots[i].shape)


if __name__ == "__main__":
    main()
