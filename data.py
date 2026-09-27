from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_npy_snapshot(path, *, device=None):
    array = np.load(Path(path)).astype(np.float32)
    return torch.as_tensor(array, device=device)


def load_intraday_log_volume(
    csv_path,
    observations_per_day: int = 78,
    column: str = "log_volume",
):
    """Convert day-major intraday data into one aggregate sample per clock bin.

    Returns array with shape:
        [observations_per_day, number_of_days]
    """
    df = pd.read_csv(csv_path)
    if column not in df:
        raise KeyError(f"missing column: {column}")

    values = (
        pd.to_numeric(df[column], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    if len(values) % observations_per_day != 0:
        raise ValueError(
            "row count must be divisible by observations_per_day"
        )

    days = len(values) // observations_per_day
    day_major = values.reshape(days, observations_per_day)
    return day_major.T
