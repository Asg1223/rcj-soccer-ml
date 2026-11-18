#!/usr/bin/env python3
"""教材の図（docs/assets/fig/）を作り直す。日本語フォント（Noto Sans CJK JP など）が必要。

    python tools/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "fig"
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Hiragino Sans", "Yu Gothic", "sans-serif"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# ルールの説明に合わせた 1 周期（833 µs）：強さ 1 → 1/4 → 1/16 → 1/64 → 休み（約 346 µs）
PERIOD = 833.0
SPACE = 346.0
STEP = (PERIOD - SPACE) / 4
LEVELS = [1, 1 / 4, 1 / 16, 1 / 64]


def envelope(t):
    t = np.mod(t, PERIOD)
    out = np.zeros_like(t)
    for k, lv in enumerate(LEVELS):
        out[(t >= k * STEP) & (t < (k + 1) * STEP)] = lv
    return out


def fig_ball_waveform():
    t = np.linspace(0, 2 * PERIOD, 4000)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.fill_between(t, envelope(t), step="mid", alpha=0.35, color="tab:orange")
    ax.plot(t, envelope(t), color="tab:orange", lw=2)
    for k, lv in enumerate(LEVELS):
        ax.text((k + 0.5) * STEP, lv + 0.04, ["1", "1/4", "1/16", "1/64"][k], ha="center", fontsize=9)
    ax.text(PERIOD - SPACE / 2, 0.08, "休み", ha="center", fontsize=9)
    ax.annotate("", xy=(PERIOD, 1.12), xytext=(0, 1.12), arrowprops=dict(arrowstyle="<->"))
    ax.text(PERIOD / 2, 1.16, "1 周期 = 約 833 µs（1.2 kHz）", ha="center", fontsize=9)
    ax.set_xlabel("時間 [µs]")
    ax.set_ylabel("光の強さ（相対）")
    ax.set_ylim(0, 1.3)
    ax.set_title("IR ボールの光（40 kHz の点滅の「強さ」を階段状に変えている）", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ball_waveform.png", dpi=150)
    plt.close(fig)


def fig_receiver_pulse():
    t = np.linspace(0, 2 * PERIOD, 4000)
    env = envelope(t)
    cases = [("近い（明るい）", 80.0), ("中くらい", 8.0), ("遠い（暗い）", 1.5)]
    fig, axes = plt.subplots(len(cases), 1, figsize=(7, 4.2), sharex=True)
    for ax, (name, gain) in zip(axes, cases):
        on = (env * gain) >= 1.0   # 受光モジュールが反応できる明るさを 1 とする
        ax.fill_between(t, on.astype(float), step="mid", color="tab:blue", alpha=0.5)
        ax.set_yticks([0, 1], ["なし", "受光"])
        ax.set_ylim(-0.1, 1.3)
        ratio = on.mean()
        ax.set_title(f"{name}：受光している時間の割合 {ratio:.0%}", fontsize=9, loc="left")
    axes[-1].set_xlabel("時間 [µs]")
    fig.suptitle("受光モジュールの出力：明るいほど「受けている時間」が長い", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "receiver_pulse.png", dpi=150)
    plt.close(fig)


def fig_gradient_descent():
    w = np.linspace(-1, 3, 200)
    loss = (w - 1.2) ** 2 + 0.3
    path = [-0.6]
    for _ in range(6):
        path.append(path[-1] - 0.35 * 2 * (path[-1] - 1.2))
    path = np.array(path)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(w, loss, color="tab:gray")
    ax.plot(path, (path - 1.2) ** 2 + 0.3, "o-", color="tab:red")
    for k, p in enumerate(path[:3]):
        ax.annotate(f"{k} 回目", (p, (p - 1.2) ** 2 + 0.3), textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set_xlabel("重み w")
    ax.set_ylabel("誤差（損失）")
    ax.set_title("勾配降下法：坂を下る向きに、少しずつ w を動かす", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "gradient_descent.png", dpi=150)
    plt.close(fig)


def fig_angle_ring():
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    a = np.linspace(-180, 180, 361)
    axes[0].plot(a, a, color="tab:blue")
    axes[0].scatter([-179, 179], [-179, 179], color="tab:red", zorder=3)
    axes[0].set_xlabel("角度 [°]")
    axes[0].set_ylabel("そのまま数字にした値")
    axes[0].set_title("数字のまま：-179° と 179° が遠い", fontsize=9)
    th = np.deg2rad(a)
    axes[1].plot(np.cos(th), np.sin(th), color="tab:blue")
    for deg in (-179, 179):
        r = np.deg2rad(deg)
        axes[1].scatter([np.cos(r)], [np.sin(r)], color="tab:red", zorder=3)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("cos")
    axes[1].set_ylabel("sin")
    axes[1].set_title("cos と sin にする：-179° と 179° がとなり", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "angle_ring.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_ball_waveform()
    fig_receiver_pulse()
    fig_gradient_descent()
    fig_angle_ring()
    print("作りました:", ", ".join(p.name for p in sorted(OUT.glob("*.png"))))
