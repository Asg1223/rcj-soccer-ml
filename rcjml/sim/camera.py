"""ロボットのカメラ画像のシミュレータ（教材 第6部）。

公開できる画像がなくても第6部を進められるように、「ロボットのカメラから見たコート」の
小さな画像（96 × 96）を、ボール・黄ゴール・青ゴール・ロボットの正解の枠と一緒に作る。

作り方は単純な絵（色のついた丸や四角）だが、実機で困ることはまねしている。
    - 照明の明るさ・色あい、画像のノイズ、ぼけ
    - ボールの影と、光の当たり方のむら
    - ロボットがボールを隠す
    - ロボットはチームごとに形も色も違う（style）。オレンジや黄色の部品を持つロボットもいる
    - コートの外の人の服など、ボールやゴールに似た色

注意: 本物の写真とは見た目がかなり違う。第6部の流れを体験するためのもので、
実機で使うモデルは、自分のカメラで撮った画像で学習し直す（6-3）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

H = W = 96
CLASSES = ["ball", "yellow_goal", "blue_goal", "robot"]
HORIZON = 30          # 地平線（コートの奥の壁の下端）の y 座標の目安
FOV_DEG = 60.0        # 横の画角（教材用の仮の値）


@dataclass
class Box:
    cls: int
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2


def robot_style(style: int) -> dict:
    """チームごとのロボットの見た目。同じ style なら同じ見た目になる。"""
    r = np.random.default_rng(10_000 + style)
    body = r.uniform(0.05, 0.45, 3)                       # 黒っぽい〜灰色の本体
    if r.random() < 0.3:
        body = r.uniform(0.5, 0.85, 3)                    # 白っぽい・明るい本体のチームもある
    accents = []
    for _ in range(r.integers(1, 4)):
        c = r.uniform(0, 1, 3)
        if r.random() < 0.25:
            c = np.array([0.95, 0.55, 0.1])               # オレンジの部品（ボールとまちがえやすい）
        accents.append(c)
    return {
        "body": body,
        "accents": accents,
        "aspect": r.uniform(0.8, 1.4),                    # 横長・縦長
        "top": r.uniform(0.15, 0.45),                     # 上の部品（基板など）の高さの割合
        "round": r.random() < 0.5,
    }


def _ground_y(dist_cm: float) -> float:
    """距離から、画像の中で「床に置いたもの」の下端の y を決める（遠いほど地平線に近い）。"""
    return HORIZON + (H - HORIZON) * (12.0 / (dist_cm + 12.0)) * 1.6


def _scale(dist_cm: float) -> float:
    """1 cm の物が、画像の中で何ピクセルに見えるか（遠いほど小さい）。"""
    return 70.0 / (dist_cm + 8.0)


def _x_from_angle(angle_deg: float) -> float:
    """ロボットから見た角度（左が +）を、画像の x にする。"""
    return W / 2 - (angle_deg / (FOV_DEG / 2)) * (W / 2)


def angle_from_x(x: float) -> float:
    """画像の x から、ロボットから見た角度（左が +）を出す（6-9）。"""
    return (W / 2 - x) / (W / 2) * (FOV_DEG / 2)


def render(rng: np.random.Generator, *, robot_styles=(0, 1, 2, 3), p_ball=0.8, p_goal=0.6,
           max_robots=2, difficulty: str = "normal", ball_at=None):
    """1 枚の画像と、写っている物の正解の枠を作る。

    ball_at=(角度, 距離) を渡すと、ボールをその場所に置く（6-9 で IR と同じボールを撮るため）。

    戻り値: (画像 uint8 の (96, 96, 3), Box のリスト, 情報の辞書)
    """
    hard = difficulty == "hard"
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W, 3), np.float32)

    # --- 背景：奥の壁と、その上（観客など）---
    wall_top = HORIZON - rng.uniform(8, 16)
    img[:] = rng.uniform(0.25, 0.6, 3)                             # 会場の奥
    for _ in range(rng.integers(2, 7)):                            # 人の服など、いろいろな色のかたまり
        cx, w = rng.uniform(0, W), rng.uniform(4, 14)
        c = rng.uniform(0, 1, 3)
        if rng.random() < (0.3 if hard else 0.15):
            c = np.array([0.95, 0.5, 0.1]) * rng.uniform(0.8, 1.1)  # オレンジの服
        m = (np.abs(xx - cx) < w / 2) & (yy < wall_top)
        img[m] = c
    img[(yy >= wall_top) & (yy < HORIZON)] = rng.uniform(0.02, 0.12)   # 黒い壁

    # --- 床：緑、少しむらと白線 ---
    floor = (yy >= HORIZON)
    green = np.array([0.1, 0.45, 0.15]) * rng.uniform(0.7, 1.2)
    tex = rng.normal(0, 0.03, (H, W, 1))
    img[floor] = (green + tex)[floor]
    if rng.random() < 0.6:
        ly = rng.uniform(HORIZON + 5, H - 5)
        slope = rng.uniform(-0.3, 0.3)
        line = floor & (np.abs(yy - (ly + slope * (xx - W / 2))) < 0.8 + (yy - HORIZON) / 40)
        img[line] = 0.85

    boxes: list[Box] = []
    things = []   # (距離, 描く関数) を、遠い順に描く

    # --- ゴール：奥の壁の前に、黄色か青の四角 ---
    if rng.random() < p_goal:
        cls = 1 if rng.random() < 0.5 else 2
        color = np.array([0.95, 0.85, 0.1]) if cls == 1 else np.array([0.1, 0.3, 0.9])
        gw = rng.uniform(18, 60)
        gx = rng.uniform(-gw / 3, W + gw / 3)
        gy1 = HORIZON + rng.uniform(0, 6)
        gy0 = gy1 - gw * rng.uniform(0.25, 0.4)

        def draw_goal(color=color, gx=gx, gw=gw, gy0=gy0, gy1=gy1, cls=cls):
            m = (np.abs(xx - gx) < gw / 2) & (yy >= gy0) & (yy < gy1)
            shade = 1.0 - 0.3 * (yy - gy0) / max(gy1 - gy0, 1)
            img[m] = (color * shade[..., None])[m]
            x0, x1 = max(gx - gw / 2, 0), min(gx + gw / 2, W)
            if x1 - x0 > 3:
                boxes.append(Box(cls, x0, max(gy0, 0), x1, gy1))

        things.append((300.0, draw_goal))

    # --- ロボット ---
    for _ in range(rng.integers(0, max_robots + 1)):
        st = robot_style(int(rng.choice(robot_styles)))
        dist = rng.uniform(15, 150)
        ang = rng.uniform(-FOV_DEG * 0.6, FOV_DEG * 0.6)

        def draw_robot(st=st, dist=dist, ang=ang):
            s = _scale(dist)
            w = 20 * s * st["aspect"]
            h = 18 * s
            cx, by = _x_from_angle(ang), _ground_y(dist)
            x0, x1, y0 = cx - w / 2, cx + w / 2, by - h
            if st["round"]:
                m = (((xx - cx) / (w / 2)) ** 2 < 1) & (yy >= y0) & (yy < by)
            else:
                m = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < by)
            img[m] = st["body"] * rng.uniform(0.8, 1.1)
            ty = y0 + h * st["top"]
            for k, c in enumerate(st["accents"]):
                ax0 = x0 + w * (0.1 + 0.8 * k / max(len(st["accents"]), 1))
                am = m & (xx >= ax0) & (xx < ax0 + w * 0.22) & (yy >= y0 + h * 0.05) & (yy < ty)
                img[am] = c
            bx0, bx1 = max(x0, 0), min(x1, W)
            if bx1 - bx0 > 2 and by > 0:
                boxes.append(Box(3, bx0, max(y0, 0), bx1, min(by, H)))

        things.append((dist, draw_robot))

    # --- ボール ---
    info = {"ball": None}
    if ball_at is not None or rng.random() < p_ball:
        if ball_at is not None:
            ang, dist = float(ball_at[0]), float(ball_at[1])
        else:
            dist = rng.uniform(6, 160)
            ang = rng.uniform(-FOV_DEG * 0.55, FOV_DEG * 0.55)
        info["ball"] = {"distance_cm": dist, "angle_deg": ang}

        def draw_ball(dist=dist, ang=ang):
            s = _scale(dist)
            r = 2.1 * s                                     # 直径 4.2 cm くらい
            cx, by = _x_from_angle(ang), _ground_y(dist)
            cy = by - r
            sh = (((xx - cx) / (r * 1.3)) ** 2 + ((yy - by) / (r * 0.35)) ** 2) < 1   # 影
            img[sh] *= 0.55
            d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / max(r * r, 1e-6)
            m = d2 < 1
            light = 1.15 - 0.45 * np.clip((yy - cy + 0.4 * (xx - cx)) / max(r, 1e-6), -1, 1)
            base = np.array([1.0, 0.45, 0.05]) * rng.uniform(0.8, 1.05)
            img[m] = np.clip(base * light[..., None], 0, 1)[m]
            x0, x1 = max(cx - r, 0), min(cx + r, W)
            if x1 - x0 > 1 and cy + r > 0:
                boxes.append(Box(0, x0, max(cy - r, 0), x1, min(cy + r, H)))

        things.append((dist, draw_ball))

    for _, draw in sorted(things, key=lambda t: -t[0]):     # 遠いものから描く（近いものが隠す）
        draw()

    # 隠れて見えなくなった物の枠を消す（中心の色がその物でなくなったら、隠れたとみなす）
    boxes = _drop_hidden(img, boxes)

    # --- 照明・カメラのくせ ---
    gain = rng.uniform(0.55, 1.35) if hard else rng.uniform(0.7, 1.2)
    tint = rng.normal(1.0, 0.12 if hard else 0.06, 3)
    img = img * gain * tint
    grad = 1.0 + rng.uniform(-0.25, 0.25) * (xx - W / 2) / W                 # 片側が暗い
    img *= grad[..., None]
    img += rng.normal(0, 0.05 if hard else 0.03, img.shape)
    if rng.random() < (0.5 if hard else 0.3):                                # ぼけ（動いているとき）
        k = int(rng.integers(2, 4))
        img = sum(np.roll(img, s, axis=1) for s in range(k)) / k
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8), boxes, info


def _drop_hidden(img, boxes):
    keep = []
    for b in boxes:
        x, y = int(np.clip(b.cx, 0, W - 1)), int(np.clip(b.cy, 0, H - 1))
        c = img[y, x]
        if b.cls == 0 and not (c[0] > 0.5 and c[2] < 0.35 and c[0] > c[1] * 1.3):
            continue
        keep.append(b)
    return keep


def make_images(n: int, *, seed: int = 0, robot_styles=(0, 1, 2, 3), difficulty: str = "normal", **kw):
    """n 枚の画像と正解の枠をまとめて作る。"""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, H, W, 3), np.uint8)
    labels, infos = [], []
    for i in range(n):
        X[i], b, info = render(rng, robot_styles=robot_styles, difficulty=difficulty, **kw)
        labels.append(b)
        infos.append(info)
    return X, labels, infos


# ---------------------------------------------------------------------------
# 学習用の正解の形
# ---------------------------------------------------------------------------


def ball_position_class(boxes) -> int:
    """6-5 の分類の正解：0 = ボールなし、1 = 左、2 = 真ん中、3 = 右。"""
    balls = [b for b in boxes if b.cls == 0]
    if not balls:
        return 0
    b = max(balls, key=lambda b: (b.x1 - b.x0))
    if b.cx < W / 3:
        return 1
    if b.cx < 2 * W / 3:
        return 2
    return 3


def centroid_grid(boxes, grid: int = 12) -> np.ndarray:
    """6-6 の FOMO 風の正解：画像を grid × grid のマスに分け、物の中心があるマスにその種類を入れる。

    戻り値は (grid, grid) の整数。0 = 何もない、1〜4 = CLASSES の順。
    """
    g = np.zeros((grid, grid), np.int64)
    cell = W / grid
    for b in sorted(boxes, key=lambda b: (b.x1 - b.x0) * (b.y1 - b.y0)):   # 小さい物を優先（あとで上書き）
        gx = int(np.clip(b.cx // cell, 0, grid - 1))
        gy = int(np.clip(b.cy // cell, 0, grid - 1))
        g[gy, gx] = b.cls + 1
    return g


def boxes_to_yolo(boxes, w: int = W, h: int = H) -> str:
    """正解の枠を YOLO 形式の文字にする（6-4。ラベル付けツールの書き出し形式と同じ）。"""
    lines = []
    for b in boxes:
        lines.append(f"{b.cls} {b.cx / w:.6f} {b.cy / h:.6f} {(b.x1 - b.x0) / w:.6f} {(b.y1 - b.y0) / h:.6f}")
    return "\n".join(lines)
