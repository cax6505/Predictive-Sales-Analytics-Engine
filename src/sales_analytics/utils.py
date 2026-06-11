import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def mode_or_unknown(series: pd.Series, unknown: str = "unknown") -> str:
    series = series.dropna().astype(str)
    if series.empty:
        return unknown
    modes = series.mode()
    if modes.empty:
        return unknown
    return str(modes.iloc[0])
