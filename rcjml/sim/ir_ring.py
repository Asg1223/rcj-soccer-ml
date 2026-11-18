"""IR センサリングのシミュレータ（教材用 v1）。

実機がなくても教材を進められるように、「ボールがこの角度・この距離にあるとき、
各センサがどんな値を出すか」を、実機で起きる“くせ”ごと作る。

難しさは3段階（difficulty）:
    easy   : くせが少ない。仕組みを理解するための練習用
    normal : 実機らしいくせが一通り入る（既定）
    hard   : 大会会場を想定。反射・相手ロボットの遮へい・外乱光が強く、取り直すたびに条件が変わる

入れている“くせ”（第1部で1つずつ説明する）:
    機体ごと（robot_seed で決まる）
        - センサの感度の個体差、取り付け角度のずれ、指向性（見える範囲の広さ）のばらつき
        - 機体の柱やキャッチャーの影：特定の方向から来た光が、反対側のセンサに届きにくい
    取り直すたび（session ごと）
        - 外乱光（照明・窓）：全体に乗る明るさと、特定の方向から入る明るさ
        - ボールの明るさ（ボールの電池の減り具合）
    1回ごと（サンプルごと）
        - 壁や物からの反射：ボールの「鏡像」が別の方向から見える
        - 相手ロボットの遮へい：ボールからの光がさえぎられる
        - ボール以外の赤外線：相手ロボットのセンサなどが、別の方向から光る
        - ボールを保持しているとき、キャッチャーの壁で隣のセンサが隠れる
        - 受光の非線形性：検出のしきい値（暗いと 0）、明るいと飽和、対数っぽい応答
        - パルス変調ボールの読み取りのばらつき、値の段階（量子化）、ときどき読み取り失敗

注意: パラメータは教材用に決めた架空の値で、特定のセンサやボールや会場を測ったものではない。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rcjml.config import RobotConfig


# ---------------------------------------------------------------------------
# パラメータ
# ---------------------------------------------------------------------------


# 光の明るさの単位：センサがぎりぎり反応できる明るさ（検出しきい値）を 1 とする
SATURATION = 800.0   # これ以上明るいと、センサの値は最大のまま（飽和）


@dataclass(frozen=True)
class BallModel:
    intensity: float  # ボールのすぐそば（0 cm）での明るさ（しきい値 = 1 とした単位）
    d0_cm: float      # 明るさが半分になる距離の目安（ボールの大きさに関係する）
    k: float          # 向きの感度の鋭さ（大きいほど正面だけ強い）
    floor: float      # 回り込む光（真後ろでも少しだけ反応する）


# 架空のパラメータ。大型ボールは明るく広く、ゴルフボールサイズは暗く鋭い、という傾向だけ再現する
BALL_MODELS = {
    "large_ir": BallModel(intensity=1500.0, d0_cm=10.0, k=2.0, floor=0.03),
    "golf_ir": BallModel(intensity=900.0, d0_cm=7.0, k=3.0, floor=0.015),
}


@dataclass(frozen=True)
class Difficulty:
    # 機体ごと
    gain_spread: float           # 感度の個体差（標準偏差、割合）
    offset_spread_deg: float     # 取り付け角度のずれ（標準偏差、度）
    k_spread: float              # 指向性のばらつき（割合）
    body_shadow: float           # 柱・キャッチャーの影の濃さ（0 = なし）
    # 取り直すたび
    ambient_mean: float          # 外乱光の平均（しきい値 = 1 とした単位）
    ambient_dir: float           # 特定の方向から入る外乱光の強さ
    ball_spread: float           # ボールの明るさのばらつき（割合）
    # 1回ごと
    reflection_prob: float       # 反射が起きる確率
    reflection_coef: tuple[float, float]   # 反射の強さの範囲
    occlusion_prob: float        # 相手ロボットにさえぎられる確率
    occlusion_keep: tuple[float, float]    # さえぎられたときに残る光の割合の範囲
    stray_prob: float            # ボール以外の赤外線（相手ロボットのセンサなど）が入る確率
    stray_level: tuple[float, float]       # その明るさの範囲（しきい値 = 1）
    # 受光
    threshold: float             # これより暗いと 0 になる（1 = 標準のセンサ）
    jitter: float                # パルス読み取りのばらつき（割合）
    levels: int                  # 値の段階の数（量子化）
    dropout_prob: float          # 1センサが読み取り失敗する確率


DIFFICULTIES = {
    "easy": Difficulty(
        gain_spread=0.05, offset_spread_deg=1.0, k_spread=0.0, body_shadow=0.0,
        ambient_mean=0.0, ambient_dir=0.0, ball_spread=0.0,
        reflection_prob=0.0, reflection_coef=(0.0, 0.0),
        occlusion_prob=0.0, occlusion_keep=(1.0, 1.0),
        stray_prob=0.0, stray_level=(0.0, 0.0),
        threshold=0.0, jitter=0.02, levels=1024, dropout_prob=0.0,
    ),
    "normal": Difficulty(
        gain_spread=0.20, offset_spread_deg=6.0, k_spread=0.30, body_shadow=0.5,
        ambient_mean=0.5, ambient_dir=1.5, ball_spread=0.18,
        reflection_prob=0.45, reflection_coef=(0.15, 0.45),
        occlusion_prob=0.12, occlusion_keep=(0.15, 0.6),
        stray_prob=0.05, stray_level=(2.0, 20.0),
        threshold=1.0, jitter=0.12, levels=64, dropout_prob=0.003,
    ),
    "hard": Difficulty(
        gain_spread=0.30, offset_spread_deg=8.0, k_spread=0.40, body_shadow=0.7,
        ambient_mean=1.0, ambient_dir=4.0, ball_spread=0.28,
        reflection_prob=0.60, reflection_coef=(0.20, 0.70),
        occlusion_prob=0.25, occlusion_keep=(0.05, 0.5),
        stray_prob=0.15, stray_level=(3.0, 60.0),
        threshold=1.5, jitter=0.18, levels=32, dropout_prob=0.008,
    ),
}

CATCH_DISTANCE_CM = 8.0   # これより近く、かつ正面 ±CATCH_ANGLE_DEG ならボール保持とみなす
CATCH_ANGLE_DEG = 20.0


def _wrap(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def _difficulty(d: str | Difficulty) -> Difficulty:
    if isinstance(d, Difficulty):
        return d
    if d not in DIFFICULTIES:
        raise ValueError(f"difficulty は {tuple(DIFFICULTIES)} のどれかにしてください")
    return DIFFICULTIES[d]


# ---------------------------------------------------------------------------
# 機体ごとの性質
# ---------------------------------------------------------------------------


@dataclass
class RobotTraits:
    gains: np.ndarray          # 感度の個体差（センサごと）
    offsets_deg: np.ndarray    # 取り付け角度のずれ（センサごと）
    k_scale: np.ndarray        # 指向性の鋭さの倍率（センサごと）
    posts_deg: np.ndarray      # 柱の方向（機体の中心から見た角度）


def robot_traits(cfg: RobotConfig, difficulty="normal", seed: int = 0) -> RobotTraits:
    """機体の個性。同じ seed なら同じ「機体」になる。"""
    d = _difficulty(difficulty)
    n = cfg.ir.sensor_count
    rng = np.random.default_rng(seed + 1_000)
    gains = np.clip(rng.normal(1.0, d.gain_spread, n), 0.4, 1.8)
    offsets = np.clip(rng.normal(0.0, d.offset_spread_deg, n), -15.0, 15.0)
    k_scale = np.clip(rng.normal(1.0, d.k_spread, n), 0.5, 1.8)
    # 柱は 3 本、キャッチャーの左右（正面 ±35° 付近）と後ろのどこか
    posts = np.array([35.0, -35.0, rng.uniform(120.0, 240.0)]) + rng.normal(0, 5, 3)
    return RobotTraits(gains, offsets, k_scale, posts)


# 互換用（v0 の関数名）
def sensor_gains(cfg: RobotConfig, seed: int = 0, difficulty="normal") -> np.ndarray:
    return robot_traits(cfg, difficulty, seed).gains


def sensor_offsets(cfg: RobotConfig, seed: int = 0, difficulty="normal") -> np.ndarray:
    return robot_traits(cfg, difficulty, seed).offsets_deg


# ---------------------------------------------------------------------------
# 取り直すたびの条件
# ---------------------------------------------------------------------------


@dataclass
class Session:
    ambient: float             # 全体に乗る外乱光
    ambient_dir_deg: float     # 外乱光が入ってくる方向
    ambient_dir: float         # その方向からの外乱光の強さ
    ball_scale: float          # ボールの明るさの倍率


def sample_session(difficulty="normal", rng: np.random.Generator | None = None) -> Session:
    d = _difficulty(difficulty)
    rng = rng or np.random.default_rng()
    return Session(
        ambient=d.ambient_mean * rng.uniform(0.3, 1.7),
        ambient_dir_deg=rng.uniform(-180.0, 180.0),
        ambient_dir=d.ambient_dir * rng.uniform(0.0, 1.5),
        ball_scale=float(np.clip(rng.normal(1.0, d.ball_spread), 0.5, 1.5)),
    )


NOMINAL_SESSION = Session(ambient=0.0, ambient_dir_deg=0.0, ambient_dir=0.0, ball_scale=1.0)


# ---------------------------------------------------------------------------
# シミュレーション本体
# ---------------------------------------------------------------------------


def _lobe(ball: BallModel, k_scale, diff_deg):
    """向きの感度。diff_deg はセンサの向きと光の来る方向のずれ。"""
    c = np.clip(np.cos(np.deg2rad(diff_deg)), 0.0, None)
    return c ** (ball.k * k_scale) + ball.floor


def _brightness(ball: BallModel, dist_cm):
    return ball.intensity / (1.0 + (np.asarray(dist_cm) / ball.d0_cm) ** 2)


def simulate(
    cfg: RobotConfig,
    angle_deg,
    distance_cm,
    *,
    difficulty="normal",
    robot_seed: int = 0,
    session: Session | None = None,
    rng: np.random.Generator | None = None,
    return_details: bool = False,
):
    """ボールの角度・距離（1つでも配列でもよい）から、センサ値の配列を返す。

    戻り値の形は (サンプル数, センサ数)。値は 0〜max_reading の整数。
    return_details=True なら、反射・遮へいが起きたかどうかも返す。
    """
    d = _difficulty(difficulty)
    rng = rng or np.random.default_rng()
    ball = BALL_MODELS[cfg.ball_type]
    traits = robot_traits(cfg, d, robot_seed)
    session = session or NOMINAL_SESSION

    angle = np.atleast_1d(np.asarray(angle_deg, dtype=float))[:, None]
    dist = np.atleast_1d(np.asarray(distance_cm, dtype=float))[:, None]
    m = angle.shape[0]
    sensors = (np.asarray(cfg.ir.angles_deg) + traits.offsets_deg)[None, :]
    k_scale = traits.k_scale[None, :]

    # --- 直接届く光 ---
    direct = _brightness(ball, dist) * _lobe(ball, k_scale, _wrap(angle - sensors))

    # 柱・キャッチャーの影：柱の方向から来た光は、その方向を向いていないセンサに届きにくい
    if d.body_shadow > 0:
        for post in traits.posts_deg:
            near_post = np.exp(-((_wrap(angle - post) / 12.0) ** 2))          # (m, 1)
            facing = np.abs(_wrap(sensors - angle)) < 40.0                     # (m, n)
            direct = direct * (1.0 - d.body_shadow * near_post * ~facing)

    # ボール保持中：キャッチャーの壁で、正面以外のセンサが隠れる
    holding = (dist[:, 0] < CATCH_DISTANCE_CM) & (np.abs(_wrap(angle[:, 0])) < CATCH_ANGLE_DEG)
    if d.body_shadow > 0 and holding.any():
        side = np.abs(_wrap(sensors - angle)) > 20.0
        direct[holding] = direct[holding] * np.where(side[holding], 0.25, 1.0)

    # 相手ロボットにさえぎられる
    occluded = rng.random(m) < d.occlusion_prob
    keep = rng.uniform(*d.occlusion_keep, m)
    direct = direct * np.where(occluded, keep, 1.0)[:, None]

    # --- 反射（ボールの鏡像） ---
    reflected = np.zeros_like(direct)
    has_reflection = rng.random(m) < d.reflection_prob
    if has_reflection.any():
        idx = np.where(has_reflection)[0]
        # ボールの位置（ロボット中心、正面 = +x、左 = +y）
        th = np.deg2rad(angle[idx, 0])
        bx, by = dist[idx, 0] * np.cos(th), dist[idx, 0] * np.sin(th)
        # 反射面：ランダムな向き・距離の直線（壁、相手ロボット、ゴールなど）
        psi = rng.uniform(-np.pi, np.pi, idx.size)
        wall_d = dist[idx, 0] * rng.uniform(0.6, 2.0) + rng.uniform(5, 40, idx.size)
        nx, ny = np.cos(psi), np.sin(psi)
        s = bx * nx + by * ny - wall_d                 # ボールから壁までの符号つき距離
        ix, iy = bx - 2 * s * nx, by - 2 * s * ny      # 鏡像の位置
        img_angle = np.rad2deg(np.arctan2(iy, ix))[:, None]
        img_dist = np.hypot(ix, iy)[:, None]
        coef = rng.uniform(*d.reflection_coef, idx.size)[:, None]
        reflected[idx] = coef * _brightness(ball, img_dist) * _lobe(
            ball, k_scale, _wrap(img_angle - sensors)
        )

    # --- ボール以外の赤外線（相手ロボットのセンサ、照明のちらつきなど） ---
    stray = np.zeros_like(direct)
    has_stray = rng.random(m) < d.stray_prob
    if has_stray.any():
        idx = np.where(has_stray)[0]
        src = rng.uniform(-180.0, 180.0, idx.size)[:, None]
        lvl = rng.uniform(*d.stray_level, idx.size)[:, None]
        stray[idx] = lvl * np.clip(np.cos(np.deg2rad(_wrap(src - sensors[0]))), 0.0, None) ** 6

    # --- 外乱光 ---
    ambient = session.ambient + session.ambient_dir * np.clip(
        np.cos(np.deg2rad(_wrap(sensors - session.ambient_dir_deg))), 0.0, None
    ) ** 2

    light = (session.ball_scale * (direct + reflected) + stray + ambient) * traits.gains[None, :]

    # --- 受光の非線形性と読み取り ---
    light = light * (1.0 + rng.normal(0.0, d.jitter, light.shape))
    light = np.clip(light, 0.0, None)
    # 対数っぽい応答：暗い光の違いには敏感、明るい光の違いには鈍く、最後は飽和する
    level = np.log1p(light) / np.log1p(SATURATION)
    level = np.where(light < d.threshold, 0.0, level)
    level = np.clip(level, 0.0, 1.0)
    level = np.round(level * (d.levels - 1)) / (d.levels - 1)

    # ときどき読み取り失敗（0 か最大値に張りつく）
    fail = rng.random(level.shape) < d.dropout_prob
    level = np.where(fail, rng.integers(0, 2, level.shape).astype(float), level)

    values = np.rint(level * cfg.ir.max_reading).astype(int)
    if return_details:
        return values, {
            "reflection": has_reflection, "occluded": occluded, "stray": has_stray, "holding": holding,
        }
    return values


def make_dataset(
    cfg: RobotConfig,
    n: int = 3000,
    *,
    difficulty="normal",
    sessions: int = 10,
    seed: int = 0,
    robot_seed: int = 0,
    distance_range: tuple[float, float] = (3.0, 180.0),
    hold_fraction: float = 0.08,
    venue: Session | None = None,
) -> pd.DataFrame:
    """ボールをランダムな位置に置いたデータを n 個作る。sessions 回に分けて「取り直す」。

    列:
        s0, s1, ...  センサ値
        angle_deg    ボールの角度（正面 0°、反時計回り +、-180〜180）
        distance_cm  ボールまでの距離
        has_ball     ボールを保持しているか（0/1）
        session      何回目の取り直しか（学習用とテスト用を分けるときに使う）
    venue を渡すと、取り直しの条件（外乱光・ボールの明るさ）を毎回ランダムに選ばず、
    その会場の条件の近くにする（7-1 の会場キャリブレーション）。
        reflection, occluded, stray  反射・遮へい・ボール以外の赤外線が起きたか
                     （答え合わせ用。モデルの入力には使わない）
    """
    rng = np.random.default_rng(seed)
    per = np.full(sessions, n // sessions)
    per[: n % sessions] += 1
    frames = []
    for sid, k in enumerate(per):
        sess = sample_session(difficulty, rng) if venue is None else Session(
            ambient=venue.ambient * rng.uniform(0.9, 1.1),
            ambient_dir_deg=venue.ambient_dir_deg + rng.normal(0.0, 8.0),
            ambient_dir=venue.ambient_dir * rng.uniform(0.9, 1.1),
            ball_scale=venue.ball_scale * rng.uniform(0.97, 1.03),
        )
        # 距離は近いほうを多めに（ボールを追いかけている時間が長いので）
        u = rng.random(k)
        lo, hi = distance_range
        dist = lo + (hi - lo) * u**1.6
        angle = rng.uniform(-180.0, 180.0, k)
        # 保持しているデータも一定の割合で混ぜる
        hold = rng.random(k) < hold_fraction
        angle[hold] = rng.uniform(-CATCH_ANGLE_DEG * 0.8, CATCH_ANGLE_DEG * 0.8, hold.sum())
        dist[hold] = rng.uniform(lo, CATCH_DISTANCE_CM * 0.9, hold.sum())
        x, info = simulate(
            cfg, angle, dist, difficulty=difficulty, robot_seed=robot_seed,
            session=sess, rng=rng, return_details=True,
        )
        df = pd.DataFrame(x, columns=[f"s{i}" for i in range(cfg.ir.sensor_count)])
        df["angle_deg"] = np.round(angle, 2)
        df["distance_cm"] = np.round(dist, 1)
        df["has_ball"] = info["holding"].astype(int)
        df["session"] = sid
        df["reflection"] = info["reflection"].astype(int)
        df["occluded"] = info["occluded"].astype(int)
        df["stray"] = info["stray"].astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 従来手法と評価
# ---------------------------------------------------------------------------


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


def make_no_ball(
    cfg: RobotConfig,
    n: int = 500,
    *,
    difficulty="normal",
    sessions: int = 5,
    seed: int = 0,
    robot_seed: int = 0,
) -> pd.DataFrame:
    """ボールが見えないとき（コートの外、相手ロボットの真後ろなど）のデータを作る（7-2）。

    センサに入るのは、外乱光とボール以外の赤外線だけになる。
    angle_deg・distance_cm は意味がないので NaN、visible = 0 にする。
    """
    rng = np.random.default_rng(seed + 50_000)
    per = np.full(sessions, n // sessions)
    per[: n % sessions] += 1
    frames = []
    for sid, k in enumerate(per):
        sess = sample_session(difficulty, rng)
        angle = rng.uniform(-180.0, 180.0, k)
        x = simulate(cfg, angle, np.full(k, 1.0e4), difficulty=difficulty, robot_seed=robot_seed,
                     session=sess, rng=rng)
        df = pd.DataFrame(x, columns=[f"s{i}" for i in range(cfg.ir.sensor_count)])
        df["angle_deg"] = np.nan
        df["distance_cm"] = np.nan
        df["has_ball"] = 0
        df["visible"] = 0
        df["session"] = sid
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
