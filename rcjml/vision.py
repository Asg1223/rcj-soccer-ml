"""第6部のカメラ認識の道具：色のしきい値（従来の方法）、FOMO 風の検出、評価。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcjml.sim.camera import CLASSES, H, W, Box, centroid_grid

# 色のしきい値（OpenCV の HSV：H は 0〜179、S と V は 0〜255）。6-2 で自分の会場に合わせて調整する
HSV_RANGES = {
    0: [((5, 120, 120), (22, 255, 255))],     # ボール：オレンジ
    1: [((22, 120, 120), (35, 255, 255))],    # 黄ゴール
    2: [((100, 120, 80), (130, 255, 255))],   # 青ゴール
}


@dataclass
class Detection:
    cls: int
    cx: float
    cy: float
    score: float = 1.0


def hsv_detect(img_rgb: np.ndarray, ranges=None, min_area: int = 2, horizon: int | None = None) -> list[Detection]:
    """色のしきい値で、ボール・ゴールを探す（6-2）。ロボットは色が決まっていないので探せない。

    horizon を渡すと、ボールは地平線より下（床の上）だけ、ゴールは地平線のまわりだけを探す。
    """
    import cv2

    ranges = ranges or HSV_RANGES
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    rows = np.arange(hsv.shape[0])[:, None]
    out = []
    for cls, rs in ranges.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in rs:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        if horizon is not None:
            ok = (rows >= horizon - 2) if cls == 0 else ((rows >= horizon - 25) & (rows <= horizon + 8))
            mask = (mask * np.broadcast_to(ok, mask.shape)).astype(np.uint8)
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        comps = [(stats[i, cv2.CC_STAT_AREA], cents[i]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        if cls == 0:
            comps = sorted(comps, key=lambda c: -c[0])[:1]      # ボールは 1 つ：いちばん大きいかたまり
        elif comps:
            comps = sorted(comps, key=lambda c: -c[0])[:1]      # ゴールも 1 つずつ
        for area, (cx, cy) in comps:
            out.append(Detection(cls, float(cx), float(cy), float(area)))
    return out


def evaluate(pred_lists, box_lists, tol: float = 4.0) -> dict:
    """種類ごとに「見つけた割合（再現率）」と「当たった割合（適合率）」を出す。

    予測の中心が、同じ種類の正解の枠の中（または tol ピクセル以内）にあれば当たり。
    正解 1 つにつき当たりは 1 つまで。
    """
    res = {}
    for cls, name in enumerate(CLASSES):
        tp = fp = fn = 0
        for preds, boxes in zip(pred_lists, box_lists):
            gts = [b for b in boxes if b.cls == cls]
            ps = [p for p in preds if p.cls == cls]
            used = set()
            for p in ps:
                hit = None
                for j, b in enumerate(gts):
                    if j in used:
                        continue
                    if b.x0 - tol <= p.cx <= b.x1 + tol and b.y0 - tol <= p.cy <= b.y1 + tol:
                        hit = j
                        break
                if hit is None:
                    fp += 1
                else:
                    used.add(hit)
                    tp += 1
            fn += len(gts) - len(used)
        res[name] = {
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "n_true": tp + fn,
        }
    return res


# ---------------------------------------------------------------------------
# FOMO 風の検出（6-6）
# ---------------------------------------------------------------------------

GRID = 12   # 96 / 8


def grid_targets(box_lists, grid: int = GRID) -> np.ndarray:
    return np.stack([centroid_grid(b, grid) for b in box_lists])


def build_fomo(n_classes: int = len(CLASSES), width: int = 16, input_shape=(H, W, 3)):
    """小さな CNN。96×96 の画像を 12×12 のマスに分け、マスごとに「何の中心があるか」を出す。"""
    import keras
    from keras import layers

    inp = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inp)
    x = layers.Conv2D(width, 3, strides=2, padding="same", activation="relu")(x)        # 48×48
    x = layers.Conv2D(width * 2, 3, strides=2, padding="same", activation="relu")(x)    # 24×24
    x = layers.Conv2D(width * 2, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(width * 4, 3, strides=2, padding="same", activation="relu")(x)    # 12×12
    x = layers.Conv2D(width * 4, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(width * 2, 1, activation="relu")(x)
    out = layers.Conv2D(n_classes + 1, 1, activation="softmax")(x)                      # 12×12×(1+4)
    return keras.Model(inp, out)


def fomo_loss(object_weight: float = 20.0):
    """マスのほとんどは「何もない」なので、物のあるマスの重みを大きくした交差エントロピー。"""
    from keras import ops

    def loss(y_true, y_pred):
        y = ops.cast(ops.reshape(y_true, (-1, GRID, GRID)), "int32")
        onehot = ops.one_hot(y, len(CLASSES) + 1)
        ce = -ops.sum(onehot * ops.log(ops.clip(y_pred, 1e-7, 1.0)), axis=-1)
        w = ops.where(y > 0, object_weight, 1.0)
        return ops.mean(ce * w)

    return loss


def fomo_detections(prob: np.ndarray, threshold: float = 0.5) -> list[Detection]:
    """1 枚分の出力（12×12×5）から、検出のリストを作る。となりあった同じ種類のマスは 1 つにまとめる。"""
    cell = W / prob.shape[1]
    cls_map = prob.argmax(-1)
    score = prob.max(-1)
    out = []
    seen = np.zeros(cls_map.shape, bool)
    for gy in range(prob.shape[0]):
        for gx in range(prob.shape[1]):
            c = cls_map[gy, gx]
            if c == 0 or score[gy, gx] < threshold or seen[gy, gx]:
                continue
            stack, cells = [(gy, gx)], []
            seen[gy, gx] = True
            while stack:                                   # となりのマスをたどる
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < prob.shape[0] and 0 <= nx < prob.shape[1] and not seen[ny, nx]
                            and cls_map[ny, nx] == c and score[ny, nx] >= threshold):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys, xs = zip(*cells)
            out.append(Detection(int(c) - 1, (np.mean(xs) + 0.5) * cell, (np.mean(ys) + 0.5) * cell,
                                 float(max(score[y, x] for y, x in cells))))
    return out


def augment_images(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """明るさ・色あい・ノイズ・一部を隠す、のデータ拡張（6-8）。正解の位置は変わらないものだけ。"""
    X = X.astype(np.float32)
    n = len(X)
    X *= rng.uniform(0.6, 1.4, (n, 1, 1, 1)) * rng.normal(1.0, 0.1, (n, 1, 1, 3))
    X += rng.normal(0, 8, X.shape)
    for i in range(n):
        if rng.random() < 0.3:                          # 四角い部分を灰色で隠す（Cutout）
            s = int(rng.integers(8, 24))
            y, x = int(rng.integers(0, H - s)), int(rng.integers(0, W - s))
            X[i, y:y + s, x:x + s] = 128
    return np.clip(X, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 自分で撮ってラベルを付けた画像を読む（6-4）
# ---------------------------------------------------------------------------


def load_yolo_folder(folder, size: int = W, class_names=None):
    """YOLO 形式でラベルを付けた画像のフォルダを読む。

    folder/
      2026-11-20_bukatsu/img_00001.jpg と img_00001.txt（YOLO 形式）…
    サブフォルダの名前を「取り直し（session）」として返す。

    class_names を渡すと、ラベル付けツールでの順番を、この教材の順番（CLASSES）に並べかえる。
    戻り値: (画像 (n, size, size, 3) uint8, Box のリストのリスト, session のリスト)
    """
    from pathlib import Path

    import cv2

    folder = Path(folder).expanduser()
    remap = None
    if class_names is not None:
        remap = {i: CLASSES.index(n) for i, n in enumerate(class_names) if n in CLASSES}
    images, labels, sessions = [], [], []
    for img_path in sorted(folder.rglob("*.jpg")) + sorted(folder.rglob("*.png")):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        boxes = []
        txt = img_path.with_suffix(".txt")
        if txt.exists():
            for line in txt.read_text().split("\n"):
                parts = line.split()
                if len(parts) != 5:
                    continue
                c = int(parts[0])
                if remap is not None:
                    if c not in remap:
                        continue
                    c = remap[c]
                cx, cy, w, h = (float(v) * size for v in parts[1:])
                boxes.append(Box(c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        images.append(img)
        labels.append(boxes)
        sessions.append(img_path.parent.name)
    return np.stack(images) if images else np.zeros((0, size, size, 3), np.uint8), labels, sessions
