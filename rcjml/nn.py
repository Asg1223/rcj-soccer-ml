"""第3部のニューラルネット「BallNet」の入出力と、学習のための道具。

BallNet は、センサの値（0〜1 にそろえたもの）から、次の 4 つを出す 1 つの MLP。
    out[0] = cos(角度)
    out[1] = sin(角度)
    out[2] = (log(距離) - DIST_MU) / DIST_SIGMA      距離を「log」にして大きさをそろえたもの
    out[3] = ボールを保持しているかの「ロジット」     シグモイドに通すと確率になる
"""

from __future__ import annotations

import numpy as np

DIST_MU = 3.4      # log(30 cm) くらい
DIST_SIGMA = 1.0


def features(df, cfg, columns=None) -> np.ndarray:
    """センサの値を 0〜1 にそろえる。"""
    from rcjml.data import sensor_columns

    cols = columns or sensor_columns(df)
    return (df[cols].values / cfg.ir.max_reading).astype(np.float32)


def targets(df) -> np.ndarray:
    th = np.deg2rad(df["angle_deg"].values)
    z = (np.log(df["distance_cm"].values) - DIST_MU) / DIST_SIGMA
    hold = df["has_ball"].values if "has_ball" in df else np.zeros(len(df))
    return np.c_[np.cos(th), np.sin(th), z, hold].astype(np.float32)


def decode(P) -> dict:
    """モデルの出力を、角度・距離・保持の確率・自信に直す。"""
    P = np.asarray(P)
    out = {
        "angle_deg": np.rad2deg(np.arctan2(P[:, 1], P[:, 0])),
        "confidence": np.hypot(P[:, 0], P[:, 1]),
    }
    if P.shape[1] > 2:
        out["distance_cm"] = np.exp(P[:, 2] * DIST_SIGMA + DIST_MU)
    if P.shape[1] > 3:
        out["p_hold"] = 1.0 / (1.0 + np.exp(-P[:, 3]))
    return out


def evaluate(P, df) -> dict:
    """角度の誤差、距離の誤差（割合）、保持の正解率などをまとめて出す。"""
    from rcjml.sim import angle_error_deg

    d = decode(P)
    r = {"angle_MAE": float(angle_error_deg(d["angle_deg"], df["angle_deg"]).mean())}
    if "distance_cm" in d:
        rel = np.abs(d["distance_cm"] - df["distance_cm"].values) / df["distance_cm"].values
        r["dist_rel_median"] = float(np.median(rel))
    if "p_hold" in d and "has_ball" in df:
        pred = d["p_hold"] > 0.5
        true = df["has_ball"].values == 1
        r["hold_acc"] = float((pred == true).mean())
        r["hold_recall"] = float(pred[true].mean()) if true.any() else float("nan")
        r["hold_false_alarm"] = float(pred[~true].mean())
    return r


def rotate_ring(df, k: int, columns=None):
    """センサリングを k 個ぶん回したデータを作る（3-5 のデータ拡張）。

    センサが等間隔なら、ボールが 360/N × k 度回ったときの値と同じになる「はず」。
    """
    from rcjml.data import sensor_columns

    cols = columns or sensor_columns(df)
    n = len(cols)
    out = df.copy()
    vals = df[cols].values
    out[cols] = np.roll(vals, k, axis=1)
    out["angle_deg"] = (df["angle_deg"].values + 360.0 * k / n + 180.0) % 360.0 - 180.0
    return out


def augment_rotations(df, columns=None):
    """全部の回し方（0〜N-1 個）を足したデータを作る。行数は約 N 倍になる。

    ボールを保持しているデータ（has_ball = 1）は、回すと「横で保持している」という
    ありえないデータになるので、回さずにそのまま入れる。
    """
    import pandas as pd
    from rcjml.data import sensor_columns

    cols = columns or sensor_columns(df)
    free = df[df.get("has_ball", 0) == 0] if "has_ball" in df else df
    parts = [df] + [rotate_ring(free, k, cols) for k in range(1, len(cols))]
    return pd.concat(parts, ignore_index=True)


def keras_loss(hold_weight: float = 1.0):
    """BallNet 用の損失：位置（cos, sin, 距離）は二乗誤差、保持はロジットの交差エントロピー。"""
    import keras
    from keras import ops

    def loss(y_true, y_pred):
        pos = ops.mean(ops.square(y_true[:, :3] - y_pred[:, :3]), axis=-1)
        hold = keras.losses.binary_crossentropy(y_true[:, 3:4], y_pred[:, 3:4], from_logits=True)
        return pos + hold_weight * ops.reshape(hold, (-1,))

    return loss


def build_ballnet(n_in: int = 12, hidden=(64, 64), n_out: int = 4, l2: float = 0.0, dropout: float = 0.0):
    """BallNet を作る。Dense と ReLU だけなので、そのままマイコンに書き出せる（第4部）。"""
    import keras

    reg = keras.regularizers.l2(l2) if l2 else None
    layers = [keras.Input(shape=(n_in,))]
    for h in hidden:
        layers.append(keras.layers.Dense(h, activation="relu", kernel_regularizer=reg))
        if dropout:
            layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(n_out, activation="linear"))
    return keras.Sequential(layers)
