"""よく使うグラフ（教材 1-5 以降）。文字は Colab で文字化けしないよう英語にしている。"""

from __future__ import annotations

import numpy as np

from rcjml.config import RobotConfig


def polar(ax, cfg: RobotConfig, values, *, true_angle=None, pred_angle=None, title=None):
    """センサ値を、ロボットを真上から見た円グラフ風に描く。正面（0°）が上、左が +。

    ax は projection="polar" で作ったもの。
    """
    theta = np.deg2rad(cfg.ir.angles_deg)
    width = 2 * np.pi / cfg.ir.sensor_count * 0.8
    vmax = cfg.ir.max_reading
    ax.set_theta_zero_location("N")
    ax.bar(theta, values, width=width, alpha=0.7, color="tab:blue")
    if true_angle is not None:
        ax.plot([np.deg2rad(true_angle)] * 2, [0, vmax], color="tab:orange", lw=3, label="true")
    if pred_angle is not None:
        ax.plot([np.deg2rad(pred_angle)] * 2, [0, vmax], color="tab:green", lw=2, ls="--", label="pred")
    ax.set_ylim(0, vmax)
    ax.set_yticklabels([])
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def polar_grid(cfg: RobotConfig, rows, *, ncols: int = 4, size: float = 3.0, titles=None,
               true_angles=None, pred_angles=None):
    """polar をたくさん並べる。rows はセンサ値の2次元配列。"""
    import matplotlib.pyplot as plt

    rows = np.atleast_2d(rows)
    n = len(rows)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows),
                             subplot_kw={"projection": "polar"}, squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.set_visible(False)
            continue
        polar(
            ax, cfg, rows[k],
            true_angle=None if true_angles is None else true_angles[k],
            pred_angle=None if pred_angles is None else pred_angles[k],
            title=None if titles is None else titles[k],
        )
    fig.tight_layout()
    return fig
