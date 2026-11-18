# 5-3 CoreS3 に表示する

## ねらい

- CoreS3 でパケットを受け取り、BallNet で推定して、画面に「ボールの方向・距離・保持」を表示できる
- センサ側のマイコンがなくても、PC からシミュレータの値を送って動作を確かめられる

---

## 理論：画面で見えると、調整が速くなる

ロボットの調整でいちばん時間がかかるのは、「今ロボットが何を見ていて、何を考えているか」がわからないことです。CoreS3 の画面に

- センサの値（棒の長さ）
- 推定したボールの方向（オレンジの矢印）と距離
- 保持しているか（ボールの色が緑になる）
- 受け取ったパケットの数と、CRC エラーの数

を出しておくと、コートの上でロボットを持ち上げなくても、何が起きているかがすぐわかります。

---

## 手を動かす 1：スケッチを用意する

使うのは `firmware/cores3/ball_display/ball_display.ino` です。まず、書き出す BallNet を学習します（4-1 と同じ。`notebooks/05_two_boards.ipynb` にも入っています）。

```python
import numpy as np
import keras
from rcjml import load_config
from rcjml.data import load_synthetic, split_by_session
from rcjml.nn import features, targets, evaluate, keras_loss, build_ballnet

cfg = load_config()
df = load_synthetic("ring12_large_normal")
train_all, test = split_by_session(df)
last = train_all.session.max()
train, val = train_all[train_all.session < last], train_all[train_all.session == last]
keras.utils.set_random_seed(0)
model = build_ballnet()
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras_loss())
model.fit(features(train, cfg), targets(train), validation_data=(features(val, cfg), targets(val)),
          epochs=400, batch_size=64, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
print(evaluate(model.predict(features(test, cfg), verbose=0), test))
```

同じフォルダに、モデルと共通のヘッダを書き出します（4-1 と同じ）。

```python
from rcjml.config import PACKAGE_ROOT
from rcjml.export import export_to_sketch

sketch = PACKAGE_ROOT / "firmware" / "cores3" / "ball_display"
for f in export_to_sketch(model, sketch, allow_public=True):
    print(f.name)
```

!!! private "自分のモデルはチーム用リポジトリへ"
    自分のデータで学習したモデルを使うときは、`ball_display` のフォルダをチーム用リポジトリにコピーして、そこに書き出します（4-1）。

## 手を動かす 2：書き込んで、PC から試す

1. ボードを **M5CoreS3** にして書き込みます
2. CoreS3 を USB でつないだまま、PC のターミナルで次を実行します。シミュレータの値が、パケットで CoreS3 に送られます

=== "Windows"

    ```powershell
    python -m rcjml.packet send-sim --port COM5
    ```

=== "Mac"

    ```bash
    python -m rcjml.packet send-sim --port /dev/cu.usbmodem1101
    ```

ボールがロボットのまわりをゆっくり回る値が送られるので、画面のオレンジの矢印が回れば成功です。`--angle 45 --distance 20` のように、決まった場所を指定することもできます。

!!! tip "USB と Port C の両方から受け取れる"
    `ball_display` は、USB と Port C の **どちらから来たパケットでも** 受け取ります。PC で試したあと、そのままセンサ側のマイコンをつなげば動きます。

## 手を動かす 3：センサ側とつなぐ（実機）

1. センサ側のマイコンに `firmware/sensor-arduino/ir_ring_uart/ir_ring_uart.ino` を書き込みます（設定は 1-2 と同じにする。先に `python -m rcjml.export headers firmware/sensor-arduino/ir_ring_uart` でヘッダをコピー）
2. 5-1 の配線で、センサ側の TX を CoreS3 の Port C につなぎます
3. ボールをロボットのまわりで動かし、画面の矢印がついてくるか確かめます

画面の右下の数字で、通信の調子がわかります。

| 表示 | 意味 | 目安 |
|---|---|---|
| `fps` | 1 秒に何回、推定できたか | 100 前後（1-2 の読み取りが 2.5 ms のため） |
| `ok` | 正しく受け取ったパケットの数 | どんどん増える |
| `crc` | CRC ではじいたパケットの数 | ほとんど増えない |

---

## チェック問題

1. 画面の矢印は回っているのに、`crc` の数がどんどん増えています。何を疑いますか。
2. 画面の書きかえを、パケットが届くたびではなく 50 ms ごとにしているのはなぜですか。

??? note "答え"
    1. 配線のノイズ（GND、線の長さ、モータの線の近く）や、速さの設定の違いです（5-2 のつまずきポイント）。
    2. 画面の書きかえは時間がかかる処理なので、届くたびに書きかえると、パケットを受け取る時間がなくなり、取りこぼしが増えるからです。

## つまずきポイント

??? failure "画面が真っ黒のまま、`ok` が 0"
    - PC から試しているなら、`--port` が CoreS3 のポートか確かめます（Arduino IDE のシリアルモニタは閉じる）
    - センサ側からなら、Port C の RX・TX を入れかえてみます（スケッチの `PORTC_RX`, `PORTC_TX`）

??? failure "矢印の向きが左右逆"
    センサの番号の付け方（反時計回り）か、画面の描き方が逆になっています。ボールを左（+90°）に置いて、矢印が画面の左を向くか確かめます。

## もっと深く

!!! deeper "画面をちらつかせない"
    `fillRect` で消してから描くと、一瞬何もない画面が見えて、ちらつきます。M5GFX（M5Unified が使っている描画ライブラリ）の **スプライト**（`M5Canvas`）に描いてから、まとめて画面に送ると、ちらつかなくなります。
