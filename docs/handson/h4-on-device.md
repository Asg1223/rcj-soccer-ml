# H4 マイコンで推論する

| 時間の目安 | 本文 | 使う機材 |
|---|---|---|
| 90 分 | [4-1](../part4-on-device/export-c.md)、[4-2](../part4-on-device/measure.md)、[4-3](../part4-on-device/quantization.md) | PC、センサ側マイコン、IR ボール |

## ゴール

- H3 で学習した BallNet を C のヘッダに書き出し、**センサ側マイコンで動かす**
- マイコンの答えが PC と **同じ** であることを確かめる
- 1 回の推論の時間と、使うメモリを測る

---

## ステップ

### 1. スケッチをチーム用リポジトリにコピーする（5 分）

自分のデータで学習したモデルは公開しないので、スケッチのフォルダごとチーム用リポジトリにコピーしてから書き出します。

- [ ] `firmware/check/ballnet_bench` と `firmware/sensor-arduino/ballnet_infer` を、チーム用リポジトリの `firmware/` にコピーした

### 2. 書き出す（10 分）

```python
import keras
from pathlib import Path
from rcjml import load_config
from rcjml.export import export_to_sketch

cfg = load_config()
team = cfg.source.parent.parent                       # チーム用リポジトリ
model = keras.models.load_model(team / "models" / "ballnet.keras", compile=False)
for name in ["ballnet_bench", "ballnet_infer"]:
    print(export_to_sketch(model, team / "firmware" / name))
```

- [ ] 2 つのフォルダに `ballnet_model.h`、`rcjml_mlp.h`、`rcjml_packet.h` ができた

### 3. PC と同じ答えになるか確かめる（20 分）

- [ ] `ballnet_bench.ino` を書き込み、シリアルモニタに `output[0]` が出た
- [ ] 同じ入力で、PC の答えを計算した

```python
import numpy as np
from rcjml.export import layers_from_keras, forward

layers = layers_from_keras(model)
n = layers[0].W.shape[0]
x = np.array([0.1 * (i % 7) for i in range(n)], np.float32)   # ballnet_bench と同じ入力
for k in range(1000):
    x[k % n] += np.float32(1e-6)                             # ベンチの中で少しずつ変えている分
print(f"PC の output[0] = {forward(layers, x[None])[0][0]:.6f}")
```

- [ ] マイコンの `output[0]` と PC の値が、**小数第 3 位まで** 同じだった

### 4. 速さとメモリを測る（15 分）

- [ ] `us / infer`（1 回の推論の時間）と `free memory`（残りの RAM）を記録した
- [ ] Arduino IDE の「フラッシュの ○○ バイトを使っています」を記録した
- [ ] 重みを 8 ビットにした版（`export_to_sketch(..., quantized=True)`）でも測り、比べた（4-3）

### 5. センサをつないで動かす（30 分）

- [ ] `ballnet_infer.ino` の設定部分（ピン、測定窓、`ACTIVE_LOW`）を、H1 の `ir_ring_reader` と **同じ** にして書き込んだ
- [ ] シリアルモニタに `BALL,時刻,角度,距離,保持,自信,µs` が出た
- [ ] H2 の床の目印に、ボールを **8 方向 × 2 距離（20 cm、60 cm）** に置いて、表示された角度と距離を記録シートに書いた

---

## 合格条件

- [ ] マイコンと PC の `output[0]` が小数第 3 位まで同じ
- [ ] 1 回の推論の時間を測った。センサの測定窓（2.5 ms）と合わせて、**1 秒に 100 回以上** 推定できる（できないときは、ネットを小さくする・8 ビットにする・マイコンを変える）
- [ ] 8 方向 × 2 距離で、角度のずれが H3 のテストと同じくらい（だいたい 2 倍以内）
- [ ] 書き出したヘッダは、チーム用リポジトリにだけある

## 記録シート

| 項目 | float | 8 ビット |
|---|---|---|
| マイコン | | |
| 1 回の推論（µs） | | |
| 残りの RAM（バイト） | | |
| フラッシュ（バイト・%） | | |
| `output[0]`（マイコン／PC） | | |

| ボールの位置 | 0° | 45° | 90° | 135° | 180° | -135° | -90° | -45° |
|---|---|---|---|---|---|---|---|---|
| 20 cm：表示された角度・距離 | | | | | | | | |
| 60 cm：表示された角度・距離 | | | | | | | | |

---

## 困ったとき

??? failure "コンパイルで「センサの数とモデルの入力の数が違います」"
    `SENSOR_PINS` の個数と、学習に使ったセンサの数が違います。`robot.yaml`、H1 のスケッチ、このスケッチのピンの数をそろえます。

??? failure "マイコンの output[0] が PC と全然違う"
    別のモデルのヘッダを書き出していないか、`ballnet_model.h` の日付を確かめます。8 ビット版なら小数第 2 位くらいまでしか合いません（4-3）。

??? failure "Uno でメモリが足りない"
    float の重みは 64-64 だと約 21 KB あり、Uno の 32 KB のフラッシュでは苦しくなります。`hidden=(32, 32)` で学習し直すか、8 ビットにします。RAM はふつう足ります（重みはフラッシュに置いています）。

??? failure "置いた場所と角度が大きくずれる"
    スケッチの設定（特に `ACTIVE_LOW` と測定窓）が H1 と違うと、入力の値が学習のときと変わります。H1 の `ir_ring_reader` の値と、このスケッチが読んでいる値を比べます。

## 発展課題

- 推論の時間を、層の大きさ（32-32、64-64、128-128）で比べ、4-2 の見積もりの式と合うか確かめる
- CoreS3 の TensorFlow Lite Micro 版（`cores3/ballnet_tflm`、4-4）でも動かし、速さを比べる
