# 4-2 メモリと速さを測る

## ねらい

- モデルが **どれくらいのメモリを使い、1 回の推論に何 µs かかるか** を見積もり、実機で測れる
- マイコンごとの違い（Arduino Uno / ESP32 / CoreS3）を、自分で表にまとめられる
- 「ロボットの制御に間に合うか」を判断できる

---

## 理論 1：マイコンの2種類のメモリ

| メモリ | 何を置くか | Arduino Uno | ESP32-S3（CoreS3） |
|---|---|---|---|
| **フラッシュ** | プログラム、変わらないデータ（**重み**） | 32 KB | 16 MB |
| **RAM** | 計算中に変わるデータ（入力、作業用の配列） | 2 KB | 512 KB ＋ PSRAM 8 MB |

BallNet（12 → 64 → 64 → 4）なら、

- 重み：5,252 個 × 4 バイト ＝ 約 **21 KB**（フラッシュ）
- 作業用：64 個 × 2 本 × 4 バイト ＝ **512 バイト**（RAM）

です。Uno ではフラッシュがかなりきつく、ESP32 なら余裕です。

## 理論 2：計算の量から時間を見積もる

1 回の推論で「かけて足す」は 5,120 回です（4-1）。1 回の「かけて足す」に何 µs かかるかは、マイコンによって大きく違います。

- Arduino Uno（8 bit、16 MHz、小数の計算回路がない）：小数のかけ算をソフトウェアでやるので遅い
- ESP32-S3（32 bit、240 MHz、小数の計算回路がある）：ずっと速い

教材の PC（NumPy）では、1 回の推論は約 15 µs でした。マイコンでは、**実際に測る** のがいちばん確かです。

## 理論 3：間に合うかどうか

1-2 の読み取りは、1 回に 2,500 µs（2.5 ms）かかります。推論がそれより十分短ければ、**センサを読むたびに推論しても、ほとんど遅くならない** ことになります。

ロボットの制御を 1 秒に 100 回（10 ms ごと）回したいなら、読み取り ＋ 推論 ＋ モータの計算が 10 ms に収まる必要があります。

---

## 手を動かす 1：PC で見積もる

```python
from rcjml.export import layers_from_keras

layers = layers_from_keras(model)
n_params = sum(L.W.size + L.b.size for L in layers)
n_mac = sum(L.W.size for L in layers)          # かけて足す回数
width = max(L.W.shape[1] for L in layers)
print(f"重みとバイアス : {n_params} 個 = {n_params * 4 / 1024:.1f} KB（float）")
print(f"かけて足す回数 : {n_mac} 回 / 1 推論")
print(f"作業用の RAM   : {width * 2 * 4} バイト")
```

ネットの大きさを変えると、これがどう変わるかを表にします。

```python
import pandas as pd
rows = []
for hidden in [(16,), (32,), (32, 32), (64, 64), (128, 128)]:
    sizes = [12, *hidden, 4]
    params = sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
    macs = sum(a * b for a, b in zip(sizes[:-1], sizes[1:]))
    rows.append({"hidden": hidden, "params": params, "flash_KB(float)": round(params * 4 / 1024, 1),
                 "flash_KB(int8)": round(params / 1024, 1), "MAC": macs})
pd.DataFrame(rows)
```

## 手を動かす 2：実機で測る

1. 4-1 の手順で、`firmware/check/ballnet_bench` のフォルダにモデルのヘッダを書き出します

    ```python
    from rcjml.config import PACKAGE_ROOT
    from rcjml.export import export_to_sketch
    export_to_sketch(model, PACKAGE_ROOT / "firmware" / "check" / "ballnet_bench", allow_public=True)
    ```

2. `ballnet_bench.ino` を書き込み、シリアルモニタ（115200）を開きます。センサはつながなくて構いません

```text
# BallNet bench
layers      : 3
us / infer  : 1234.5
free memory : 1234
output[0]   : 0.123456
```

`us / infer` が 1 回の推論にかかった時間、`free memory` が残りの RAM です。

!!! example "やってみよう：自分の表を作る"
    持っているマイコンで測って、次のような表を作りましょう。チームのノートに残しておくと、マイコン選びやネットの大きさを決めるときに役立ちます。

    | マイコン | ネット | 重み | 1 回の推論 | 残りの RAM |
    |---|---|---|---|---|
    | （例）Arduino Uno | 32-32 float | | µs | バイト |
    | （例）ESP32 | 64-64 float | | µs | バイト |
    | （例）CoreS3 | 64-64 float | | µs | バイト |

    Arduino IDE で書き込むと、最後に「スケッチがフラッシュの ○○ バイト（○%）を使っています」と出ます。これも記録します。

---

## チェック問題

1. 重みはフラッシュと RAM のどちらに置きますか。それはなぜですか。
2. 隠れ層を 64 → 128 にすると、かけて足す回数はおよそ何倍になりますか（12 → h → h → 4 で考える）。
3. 読み取りに 2.5 ms、推論に 3 ms かかるとき、1 秒に何回ボールの位置を更新できますか。

??? note "答え"
    1. フラッシュです。重みは推論中に変わらないので、RAM（特に Uno では 2 KB しかない）を使う必要がないからです。
    2. 約 3.5 倍です（64 のとき 12×64 + 64×64 + 64×4 = 5,120、128 のとき 12×128 + 128×128 + 128×4 = 18,432）。真ん中の層が「幅 × 幅」なので、幅を 2 倍にすると、そこは 4 倍になります。
    3. 1 秒 ÷ 5.5 ms ≒ 約 180 回です。

## つまずきポイント

??? failure "`free memory` がとても少ない（Uno）"
    `Serial.print("...")` の文字列も RAM を使います。`Serial.print(F("..."))` のように `F()` で囲むと、文字列をフラッシュに置けます。

??? failure "時間が 0 µs や、ありえないほど短い"
    コンパイラが「結果を使っていない計算」を省略していることがあります。`ballnet_bench` では、入力を毎回少し変えて、最後に出力を表示することで、省略されないようにしています。

## もっと深く

!!! deeper "速くする方法"
    - **小さいネット**：成績と速さのバランスを、検証用データで確かめながら決めます
    - **int8 にする**（4-3）：Uno では整数の計算のほうが小数より速いので、重みだけでなく計算も整数にすると大きく速くなります
    - **ESP32-S3 のベクトル命令**：ESP32-S3 には、一度にたくさんのかけ算をする命令があります。Espressif の ESP-NN や ESP-DL のライブラリはこれを使っていて、TensorFlow Lite Micro（4-4）からも使えます
    - **推論を毎回しない**：ボールの位置は 1 ms で大きくは変わらないので、2 回に 1 回だけ推論する、という割り切りもあります
