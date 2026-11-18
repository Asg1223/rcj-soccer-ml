"""2 次元のコートの簡易シミュレータ（教材 5-5、7-3）。

ロボットは向きを変えずに（いつも相手ゴールの方を向いて）全方向に動ける、オムニホイールの機体とする。
ボールはロボットがキャッチャーで捕まえると一緒に動き、相手ゴールに入ればゴール。

座標（cm）：コートの中心が (0, 0)。相手ゴールは +y 側。
ボールの角度は、ロボットから見て正面（+y）が 0°、左（-x）が +。1-4 の決まりと同じ。

注意: コートやゴールの大きさは教材用の目安で、ルールの正式な寸法ではない。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FIELD_W = 180.0      # 横（x）
FIELD_H = 240.0      # 縦（y）
GOAL_W = 60.0
CATCH_DIST = 8.0     # これより近く、正面 ±CATCH_ANGLE なら捕まえる
CATCH_ANGLE = 20.0


def rel_ball(robot_xy, ball_xy):
    """ロボットから見たボールの角度（度）と距離（cm）。"""
    dx, dy = ball_xy[0] - robot_xy[0], ball_xy[1] - robot_xy[1]
    angle = np.rad2deg(np.arctan2(-dx, dy))   # 正面 +y が 0°、左（-x）が +
    return float(angle), float(np.hypot(dx, dy))


def move_vector(direction_deg: float, speed: float):
    """ロボットから見た「進む向き」（正面 0°、左 +）を、コートの x, y の速度にする。"""
    th = np.deg2rad(direction_deg)
    return np.array([-np.sin(th), np.cos(th)]) * speed


@dataclass
class Episode:
    robot: np.ndarray
    ball: np.ndarray
    holding: bool = False
    goal: bool = False
    out: bool = False
    steps: int = 0
    path: list = field(default_factory=list)


def new_episode(rng: np.random.Generator) -> Episode:
    robot = np.array([rng.uniform(-60, 60), rng.uniform(-100, -40)])
    ball = np.array([rng.uniform(-70, 70), rng.uniform(-90, 80)])
    return Episode(robot=robot, ball=ball)


def step(ep: Episode, direction_deg: float, speed: float, dt: float = 0.05) -> Episode:
    """1 ステップ進める。speed は cm/s（0〜MAX_SPEED）。"""
    v = move_vector(direction_deg, min(speed, MAX_SPEED))
    ep.robot = ep.robot + v * dt
    ep.robot = np.clip(ep.robot, [-FIELD_W / 2, -FIELD_H / 2], [FIELD_W / 2, FIELD_H / 2])
    a, d = rel_ball(ep.robot, ep.ball)
    if not ep.holding and d < CATCH_DIST and abs(a) < CATCH_ANGLE:
        ep.holding = True
    elif not ep.holding and d < CATCH_DIST:
        # キャッチャー以外に当たったら、ボールが少しはじかれる
        push = (ep.ball - ep.robot) / max(d, 1e-6)
        ep.ball = ep.ball + push * (CATCH_DIST - d + 1.0)
    if ep.holding:
        ep.ball = ep.robot + np.array([0.0, CATCH_DIST * 0.6])
    if ep.ball[1] >= FIELD_H / 2 and abs(ep.ball[0]) < GOAL_W / 2:
        ep.goal = True
    elif abs(ep.ball[0]) > FIELD_W / 2 or abs(ep.ball[1]) > FIELD_H / 2:
        ep.out = True
    ep.steps += 1
    ep.path.append((ep.robot.copy(), ep.ball.copy()))
    return ep


MAX_SPEED = 100.0    # cm/s


def chase_rule(angle_deg: float, distance_cm: float, holding: bool = False, goal_angle_deg: float = 0.0,
               k_around: float = 1.0):
    """ボールの後ろに回りこんでから進む、よくある動き方（5-5）。戻り値は（進む向き、速さ）。

    - 保持しているとき：ゴールの方向へ
    - ボールが正面付近：そのままボールへ
    - それ以外：ボールの方向より外側へずらして、回りこむ。近いほど大きく回りこむ
    """
    if holding:
        return goal_angle_deg, MAX_SPEED
    a = angle_deg
    if abs(a) < 15:
        return a, MAX_SPEED
    near = float(np.clip(40.0 / max(distance_cm, 1.0), 0.3, 1.0))
    offset = np.sign(a) * min(abs(a), 90.0) * near * k_around
    return float((a + offset + 180) % 360 - 180), MAX_SPEED * (0.7 if distance_cm < 25 else 1.0)


def goal_angle(robot_xy) -> float:
    """ロボットから見た相手ゴールの真ん中の角度（第6部ではカメラで見つける）。"""
    return rel_ball(robot_xy, np.array([0.0, FIELD_H / 2]))[0]


def run_episode(policy, rng: np.random.Generator, max_steps: int = 400, observe=None) -> Episode:
    """policy(obs) → (向き, 速さ) で 1 試合（1 本）動かす。

    observe(ep) は、ロボットが「見る」もの（角度・距離など）を返す関数。
    省略すると、本当の角度・距離（まちがいのない情報）をそのまま渡す。
    obs は辞書で、angle_deg・distance_cm・holding・goal_angle_deg を入れる。
    """
    ep = new_episode(rng)
    for _ in range(max_steps):
        if observe is None:
            a, d = rel_ball(ep.robot, ep.ball)
            obs = {"angle_deg": a, "distance_cm": d, "holding": ep.holding, "goal_angle_deg": goal_angle(ep.robot)}
        else:
            obs = observe(ep)
        direction, speed = policy(obs)
        step(ep, direction, speed)
        if ep.goal or ep.out:
            break
    return ep


def rollout(policy, rng: np.random.Generator, teacher=None, max_steps: int = 400):
    """1 本動かして、見たもの（obs）と、そのときの「お手本の動き」を記録する（7-3）。

    teacher を省略すると、policy 自身の動きを記録する（人や規則のお手本を集めるとき）。
    teacher を渡すと、動かすのは policy、記録するのは teacher の答え（DAgger）。
    戻り値は (Episode, obs のリスト, (向き, 速さ) のリスト)。
    """
    ep = new_episode(rng)
    observations, actions = [], []
    for _ in range(max_steps):
        a, d = rel_ball(ep.robot, ep.ball)
        obs = {"angle_deg": a, "distance_cm": d, "holding": ep.holding, "goal_angle_deg": goal_angle(ep.robot)}
        direction, speed = policy(obs)
        observations.append(obs)
        actions.append(teacher(obs) if teacher is not None else (direction, speed))
        step(ep, direction, speed)
        if ep.goal or ep.out:
            break
    return ep, observations, actions


def success_rate(policy, n: int = 200, seed: int = 0, observe=None, max_steps: int = 400) -> dict:
    rng = np.random.default_rng(seed)
    eps = [run_episode(policy, rng, max_steps=max_steps, observe=observe) for _ in range(n)]
    goals = [e for e in eps if e.goal]
    return {
        "goal_rate": len(goals) / n,
        "out_rate": sum(e.out for e in eps) / n,
        "mean_steps_to_goal": float(np.mean([e.steps for e in goals])) if goals else float("nan"),
    }
