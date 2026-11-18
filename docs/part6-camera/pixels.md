# 6-1 画像はただの数字の表

## ねらい

- カメラの画像が「数字の表」であることを、実際に数字を見て説明できる
- RGB と HSV の違いがわかり、なぜ色で物を探すときに HSV を使うのかを説明できる
- 解像度とフレームレートが、マイコンの計算量にどう関係するかがわかる

---

## 理論 1：画素と RGB

カメラの画像は、小さな点（**画素**、ピクセル）を縦横に並べたものです。1 つの画素は、**赤（R）・緑（G）・青（B）** の 3 つの明るさ（それぞれ 0〜255）でできています。

| 画素の色 | R | G | B |
|---|---|---|---|
| 黒 | 0 | 0 | 0 |
| 白 | 255 | 255 | 255 |
| オレンジ（ボール） | 250 | 110 | 10 |
| 黄（ゴール） | 240 | 215 | 25 |

縦 96 × 横 96 の画像なら、96 × 96 × 3 ＝ **27,648 個の数字** です。第3部の BallNet の入力は 12 個でしたから、ずっと多いですね。

## 理論 2：HSV：色を「色あい・あざやかさ・明るさ」で表す

RGB で「オレンジ」を探そうとすると、困ることがあります。同じボールでも、**照明が暗いと R・G・B が全部小さくなる** ので、「R が 200 以上」のような決め方では見つけられなくなります。

そこで、色を次の 3 つで表す **HSV** を使います。

| | 意味 | オレンジのボール |
|---|---|---|
| **H**（色相） | 色あい。赤 → 黄 → 緑 → 青 → 紫 → 赤 と一周する角度 | 暗くても明るくても、ほぼ同じ |
| **S**（彩度） | あざやかさ。0 なら灰色 | 高い |
| **V**（明度） | 明るさ | 照明で大きく変わる |

明るさの変化はほとんど **V** に入るので、**H と S** で色を決めれば、照明が変わっても見つけやすくなります。

!!! note "OpenCV の HSV の範囲"
    OpenCV では、H が 0〜179（360° を半分にしたもの）、S と V が 0〜255 です。ほかのソフトでは H が 0〜360 のこともあるので、しきい値を写すときは注意します。

## 理論 3：解像度とフレームレート

| | 意味 | 大きいと |
|---|---|---|
| **解像度** | 画像の縦 × 横の画素の数 | 遠くの小さなボールも見える。でも計算が増える |
| **フレームレート** | 1 秒に何枚撮るか（fps） | 速い動きについていける。でも 1 枚にかけられる時間が減る |

CoreS3 のカメラ（GC0308）は最大 640 × 480 ですが、マイコンで物体検出をするなら、**96 × 96 くらいに縮めて** 使うのがふつうです。画素の数が 1/33 になるので、計算も大きく減ります。

---

## 手を動かす 1：画像を作って、数字として見る

実機のカメラがなくても進められるように、教材のシミュレータで **ロボットのカメラから見たコート** の小さな画像を作ります。

=== "Colab / ローカル"

    `notebooks/06_camera.ipynb` に、第6部のコードが入っています。ローカルでは `pip install opencv-python` も実行しておきます（Colab には入っています）。

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asg1223/rcj-soccer-ml/blob/main/notebooks/06_camera.ipynb)

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rcjml.sim.camera import make_images, CLASSES

X, labels, infos = make_images(8, seed=1)
print(X.shape, X.dtype)          # (8, 96, 96, 3) uint8：8 枚、縦 96、横 96、RGB の 3 つ

COLORS = ["orange", "gold", "blue", "red"]
def show(images, box_lists, ncols=4):
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for ax, img, boxes in zip(axes.flat, images, box_lists):
        ax.imshow(img)
        for b in boxes:
            ax.add_patch(patches.Rectangle((b.x0, b.y0), b.x1 - b.x0, b.y1 - b.y0,
                                           fill=False, color=COLORS[b.cls], lw=1.5))
        ax.axis("off")
    plt.tight_layout()
    plt.show()

show(X, labels)
```

色の枠は「正解」です。オレンジがボール、金色が黄ゴール、青が青ゴール、赤がロボットです。

!!! warning "シミュレータの画像は本物とは違う"
    この画像は、色のついた丸や四角でできた簡単な絵です。第6部の **流れ** を体験するためのもので、本物のカメラの画像とは見た目がかなり違います。実機で使うモデルは、自分のカメラで撮った画像で作り直します（6-3）。

## 手を動かす 2：ボールの画素の数字を見る

```python
img, boxes = X[0], labels[0]
balls = [b for b in boxes if b.cls == 0]
if balls:
    b = balls[0]
    x, y = int(b.cx), int(b.cy)
    print("ボールの真ん中の画素 (R, G, B):", img[y, x])
print("床の画素 (R, G, B):", img[90, 10])
```

## 手を動かす 3：HSV にして、明るさを変えても H が変わらないことを確かめる

```python
import cv2

bright = X[0]
dark = (X[0] * 0.5).astype(np.uint8)          # 照明が半分になったつもり
for name, im in [("bright", bright), ("dark", dark)]:
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    if balls:
        print(name, "RGB:", im[y, x], " HSV:", hsv[y, x])
```

暗くすると RGB は全部半分くらいになりますが、HSV の **H はほとんど変わらず**、V が下がるだけです。

## 手を動かす 4：解像度で計算量がどう変わるか

```python
for w, h in [(640, 480), (320, 240), (160, 120), (96, 96)]:
    print(f"{w}×{h}: 画素 {w * h:7d} 個、数字 {w * h * 3:8d} 個、640×480 の {w * h / (640 * 480):.3f} 倍")
```

!!! example "やってみよう"
    1. `make_images(8, seed=2, difficulty="hard")` で、照明やぼけの強い画像を作って見比べましょう。
    2. 床（緑）の画素と、ロボットの画素を HSV にして比べましょう。ロボットの色は決まっているでしょうか。

---

## チェック問題

1. 160 × 120 の RGB 画像は、いくつの数字でできていますか。
2. 照明が暗くなると、HSV のどれがいちばん大きく変わりますか。
3. 解像度を縦横それぞれ半分にすると、画素の数は何分の 1 になりますか。

??? note "答え"
    1. 160 × 120 × 3 = 57,600 個です。
    2. V（明度）です。
    3. 1/4 です。

## つまずきポイント

??? failure "画像の色がおかしい（青とオレンジが入れかわる）"
    OpenCV は、画像を **BGR**（青・緑・赤の順）で扱うのがふつうです。この教材の画像は RGB なので、変換には `cv2.COLOR_RGB2HSV` を使います。`COLOR_BGR2HSV` を使うと、赤と青が入れかわります。

## もっと深く

!!! deeper "カメラの自動調整に注意"
    多くのカメラは、明るさ（**自動露出**）や色あい（**オートホワイトバランス**）を自動で変えます。人が見るにはきれいですが、色のしきい値を使うときは、「さっきと同じボールなのに H が変わった」原因になります。ESP32 のカメラのライブラリ（`esp_camera`）では、`set_whitebal` や `set_exposure_ctrl` でこれを止められます。機械学習で認識するときも、学習用の画像を撮ったときと本番で設定をそろえておくのが大事です。
