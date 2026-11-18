"""データを読み込む・分けるための小さな道具（教材 1-5、第2部）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rcjml.config import PACKAGE_ROOT, RobotConfig

SYNTHETIC_DIR = PACKAGE_ROOT / "data" / "ir-synthetic"


def load_synthetic(name: str = "ring12_large_normal") -> pd.DataFrame:
    """教材用の合成データを読む。name は data/ir-synthetic/ のファイル名（.csv なし）。"""
    path = SYNTHETIC_DIR / f"{name}.csv"
    if not path.exists():
        names = ", ".join(sorted(p.stem for p in SYNTHETIC_DIR.glob("*.csv")))
        raise FileNotFoundError(f"{path} がありません。使える名前: {names}")
    return pd.read_csv(path)


def load_log(cfg: RobotConfig, name: str = "ir_log.csv") -> pd.DataFrame:
    """rcjml.logger で集めた実機のデータを読む（設定ファイルの data.dir から）。"""
    return pd.read_csv(cfg.data_dir / name)


def sensor_columns(df: pd.DataFrame) -> list[str]:
    """s0, s1, ... の列名を番号順に返す。"""
    cols = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def split_by_session(df: pd.DataFrame, test_fraction: float = 0.25):
    """取り直し（session）の後ろのほうをテスト用にする。同じ取り直しが両方に入らない。"""
    sessions = np.sort(df["session"].unique())
    n_test = max(1, int(round(len(sessions) * test_fraction)))
    test_sessions = set(sessions[-n_test:])
    is_test = df["session"].isin(test_sessions)
    return df[~is_test].copy(), df[is_test].copy()
