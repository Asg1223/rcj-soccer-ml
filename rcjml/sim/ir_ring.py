"""IR センサリングのシミュレータ（教材用・簡易版 v0）。

実機がなくても教材を進められるように、「ボールがこの角度・この距離にあるとき、
各センサがどんな値を出すか」をそれらしく作る。

モデル（第1部で中身を説明する）:
    値_i = 上限 × 明るさ(距離) × 向きの感度(ずれの角度) × 個体差_i + ノイズ

    - 明るさ(距離)      : 遠いほど暗くなる。 1 / (1 + (d / d0)^2)
    - 向きの感度(ずれ)  : センサの正面ほど強い。 max(0, cos(ずれ))^k ＋ 床（回り込む光）
    - 個体差_i          : センサごとに感度が少しずつ違い、取り付け角度も少しずれる（実機でも必ずある）
    - ノイズ            : 毎回少しずつ値がばらつく

注意: パラメータは教材用に決めた架空の値で、特定のセンサやボールを測ったものではない。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rcjml.config import RobotConfig


@dataclass(frozen=True)
class BallModel:
    d0_cm: float      # 明るさが半分になる距離の目安
    k: float          # 向きの感度の鋭さ（大きいほど正面だけ強い）
    floor: float      # 回り込む光（真後ろでも少しだけ反応する）


# 架空のパラメータ。大型ボールは明るく広く、ゴルフボールサイズは暗く鋭い、という傾向だけ再現する
BALL_MODELS = {
    "large_ir": BallModel(d0_cm=60.0, k=2.0, floor=0.04),
    "golf_ir": BallModel(d0_cm=35.0, k=3.0, floor=0.02),
}


def _wrap(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def sensor_gains(cfg: RobotConfig, seed: int = 0, spread: float = 0.12) -> np.ndarray:
    """センサの個体差（感度のばらつき）。同じ seed なら同じ「機体」になる。"""
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(1.0, spread, cfg.ir.sensor_count), 0.6, 1.4)


def sensor_offsets(cfg: RobotConfig, seed: int = 0, spread_deg: float = 4.0) -> np.ndarray:
    """センサの取り付け角度のずれ（度）。同じ seed なら同じ「機体」になる。"""
    rng = np.random.default_rng(seed + 10_000)
    return np.clip(rng.normal(0.0, spread_deg, cfg.ir.sensor_count), -10.0, 10.0)


def simulate(
    cfg: RobotConfig,
    angle_deg,
    distance_cm,
    *,
    gains: np.ndarray | None = None,
    offsets_deg: np.ndarray | None = None,
    noise: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """ボールの角度・距離（1つでも配列でもよい）から、センサ値の配列を返す。

    戻り値の形は (サンプル数, センサ数)。値は 0〜max_reading の整数。
    """
    rng = rng or np.random.default_rng()
    ball = BALL_MODELS[cfg.ball_type]
    angle = np.atleast_1d(np.asarray(angle_deg, dtype=float))[:, None]
    dist = np.atleast_1d(np.asarray(distance_cm, dtype=float))[:, None]
    sensors = np.asarray(cfg.ir.angles_deg)[None, :]
    if offsets_deg is not None:
        sensors = sensors + np.asarray(offsets_deg)[None, :]
    if gains is None:
        gains = np.ones(cfg.ir.sensor_count)

    diff = np.deg2rad(_wrap(angle - sensors))
    direction = np.clip(np.cos(diff), 0.0, None) ** ball.k + ball.floor
    brightness = 1.0 / (1.0 + (dist / ball.d0_cm) ** 2)
    value = brightness * direction * gains[None, :]
    value = value + rng.normal(0.0, noise, value.shape)
    value = np.clip(value, 0.0, 1.0) * cfg.ir.max_reading
    return np.rint(value).astype(int)


def make_dataset(
    cfg: RobotConfig,
    n: int = 2000,
    *,
    seed: int = 0,
    robot_seed: int = 0,
    distance_range: tuple[float, float] = (10.0, 150.0),
) -> pd.DataFrame:
    """ランダムな位置にボールを置いたデータを n 個作る。

    列: s0, s1, ...（センサ値）, angle_deg（-180〜180）, distance_cm
    """
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-180.0, 180.0, n)
    dist = rng.uniform(*distance_range, n)
    gains = sensor_gains(cfg, seed=robot_seed)
    offsets = sensor_offsets(cfg, seed=robot_seed)
    x = simulate(cfg, angle, dist, gains=gains, offsets_deg=offsets, rng=rng)
    df = pd.DataFrame(x, columns=[f"s{i}" for i in range(cfg.ir.sensor_count)])
    df["angle_deg"] = np.round(angle, 2)
    df["distance_cm"] = np.round(dist, 1)
    return df


def baseline_angle(cfg: RobotConfig, readings) -> np.ndarray:
    """従来手法（重み付きベクトル和）でボールの角度を推定する。第2部のベースライン。"""
    r = np.atleast_2d(np.asarray(readings, dtype=float))
    a = np.deg2rad(np.asarray(cfg.ir.angles_deg))
    x = (r * np.cos(a)).sum(axis=1)
    y = (r * np.sin(a)).sum(axis=1)
    return np.rad2deg(np.arctan2(y, x))


def angle_error_deg(pred, true) -> np.ndarray:
    """角度の誤差（359° と 1° の差は 2°、と数える）。"""
    return np.abs(_wrap(np.asarray(pred) - np.asarray(true)))
