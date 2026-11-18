#!/usr/bin/env python3
"""教材用の合成データ（data/ir-synthetic/）を作り直す。

使い方:
    python tools/make_synthetic.py

同じバージョンの rcjml なら、何度実行しても同じデータになる（seed を固定しているため）。
"""

from pathlib import Path

from rcjml import example_config, load_config
from rcjml.sim import make_dataset

OUT = Path(__file__).resolve().parent.parent / "data" / "ir-synthetic"

# (ファイル名, センサの数, ボール, 難しさ, サンプル数, 取り直しの回数)
DATASETS = [
    ("ring12_large_easy.csv", 12, "large_ir", "easy", 3000, 6),
    ("ring12_large_normal.csv", 12, "large_ir", "normal", 6000, 12),
    ("ring12_large_hard.csv", 12, "large_ir", "hard", 12000, 16),
    ("ring12_golf_normal.csv", 12, "golf_ir", "normal", 6000, 12),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, n_sensors, ball, diff, n, sessions in DATASETS:
        cfg = load_config(example_config(n_sensors))
        cfg.ball_type = ball
        df = make_dataset(cfg, n=n, difficulty=diff, sessions=sessions, seed=2026, robot_seed=0)
        df.to_csv(OUT / name, index=False)
        print(f"{name}: {len(df)} 行, {(OUT / name).stat().st_size // 1000} KB")


if __name__ == "__main__":
    main()
