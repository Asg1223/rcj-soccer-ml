# 6-7 CoreS3 で速く動かす

## ねらい

- 物体検出のモデルの **計算量**（かけて足す回数）を数えられる
- 解像度・モデルの幅・入力の色の数が、速さと成績にどう効くかがわかる
- CoreS3 で、1 枚あたりの時間（前処理・推論）を測って、改善できる

---

## 理論 1：画像のモデルは計算が多い

6-6 の FOMO 風のモデルは、1 枚の推論で **約 1,700 万回** の「かけて足す」をします。BallNet（5,120 回）の 3,000 倍以上です。

畳み込み層 1 つの計算量は、

$$
\text{出力の縦} \times \text{出力の横} \times \text{窓の大きさ（3×3）} \times \text{入力の枚数} \times \text{出力の枚数}
$$

です。画像が大きいほど、層の「枚数（チャンネル）」が多いほど、計算が増えます。

## 理論 2：速くする方法

| 方法 | 効き方 | 気をつけること |
|---|---|---|
| **解像度を下げる** | 縦横半分で、計算は約 1/4 | 遠くの小さなボールが見えなくなる |
| **モデルの幅（チャンネル数）を減らす** | 半分で、計算は約 1/4 | 覚えられることが減り、成績が下がる |
| **見る場所をしぼる** | 地平線より上を切れば、その分減る | ゴールは地平線の近くにあるので、切りすぎない |
| **毎回は推論しない** | 2 回に 1 回なら半分 | 速い動きへの反応が遅れる |
| **グレースケールにする** | 入力が 1/3 になる（最初の層だけ） | **ボールとゴールは色で見分けているので、成績が大きく落ちる** |

最後の「グレースケール」は、一般の画像認識ではよく使われますが、RCJ のように **色に意味がある** ときは向いていません。

## 理論 3：前処理の時間も測る

マイコンでは、推論そのものだけでなく、

- カメラから画像を受け取る
- 320 × 240 を 96 × 96 に縮める
- 0〜255 の数をモデルの入力に入れる

といった **前処理** にも時間がかかります。`camera_detect` は、画面の左上に `prep`（前処理）と `infer`（推論）の時間を表示します。

---

## 手を動かす 1：計算量を数える

学習しなくても、モデルを作るだけで計算量は数えられます。

```python
import keras
import pandas as pd
from rcjml.vision import build_fomo

def count_macs(model):
    macs = 0
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            kh, kw, cin, cout = layer.kernel.shape
            oh, ow = layer.output.shape[1:3]
            macs += oh * ow * kh * kw * cin * cout
    return macs

rows = []
for size in [64, 96, 128]:
    for width in [8, 16, 32]:
        m = build_fomo(width=width, input_shape=(size, size, 3))
        rows.append({"input": f"{size}x{size}", "width": width,
                     "params": m.count_params(), "MACs (millions)": round(count_macs(m) / 1e6, 1),
                     "int8 size (KB, approx)": round(m.count_params() / 1024, 1)})
pd.DataFrame(rows)
```

6-6 で使ったのは「96 × 96、幅 16」です。幅を 8 にすると計算は約 1/4、32 にすると約 4 倍になります。

## 手を動かす 2：幅を変えて、成績と比べる

6-6 の続きです（`X_train` などが必要です）。幅 8 のモデルを学習して、6-6 の幅 16 と比べます。

```python
keras.utils.set_random_seed(0)
small = build_fomo(width=8)
small.compile(optimizer=keras.optimizers.Adam(2e-3), loss=fomo_loss())
small.fit(X_train, grid_targets(L_train), validation_data=(X_val, grid_targets(L_val)),
          epochs=40, batch_size=64, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
r_small = evaluate([fomo_detections(p, 0.5) for p in small.predict(X_test, verbose=0)], L_test)
for cls in CLASSES:
    print(f"{cls:12s} 幅16: 見つけた {r[cls]['recall']:.0%}   幅8: 見つけた {r_small[cls]['recall']:.0%}")
print("計算量:", count_macs(fomo) // 10**6, "→", count_macs(small) // 10**6, "百万回")
```

## 手を動かす 3：CoreS3 で測る（実機）

`camera_detect` を書き込み、画面の `prep` と `infer` の時間を記録します。4-2 と同じように、表にまとめておきましょう。

| モデル | 入力 | prep | infer | 1 秒に何枚 |
|---|---|---|---|---|
| 幅 16 | 96 × 96 | ms | ms | 枚 |
| 幅 8 | 96 × 96 | ms | ms | 枚 |

!!! tip "速くするための CoreS3 側の工夫"
    - カメラの解像度を最初から小さくする（`FRAMESIZE_QQVGA` = 160 × 120）と、受け取りと縮める時間が減ります
    - `modelSetInput` を 27,648 回呼ぶのは時間がかかります。ライブラリが入力の配列を直接さわれる関数を持っていれば、まとめて書きこむほうが速くなります
    - 推論の時間が長いときは、IR（第4部、1 回数 ms 以下）を毎回、カメラは数回に 1 回、のように分けます（6-9）

---

## チェック問題

1. 入力を 96 × 96 から 128 × 128 にすると、計算量はおよそ何倍になりますか。
2. RCJ のカメラ認識で、グレースケールにするのが向いていないのはなぜですか。
3. カメラの推論に 100 ms かかるとき、IR の推定（数 ms）と同じ速さで動きを決めるには、どうすればよいですか。

??? note "答え"
    1. 縦横それぞれ 4/3 倍なので、約 1.8 倍（16/9）です。
    2. ボールはオレンジ、ゴールは黄と青と、色で見分けるのがいちばん確かだからです。色の情報を捨てると、黄ゴールと青ゴールの区別もつきにくくなります。
    3. 動きは IR の推定で毎回決め、カメラの結果（ゴールの方向、相手ロボット）は届いたときだけ更新して使います（6-9）。

## もっと深く

!!! deeper "深さ方向に分ける畳み込み"
    スマートフォンやマイコン向けのモデル（MobileNet など）は、ふつうの畳み込みを「チャンネルごとの 3×3」と「1×1 でチャンネルを混ぜる」の 2 つに分けた **深さ方向分離畳み込み**（depthwise separable convolution）を使って、計算を 1/8 くらいに減らしています。Keras では `layers.SeparableConv2D` や `DepthwiseConv2D` で書けます。`build_fomo` の畳み込みを置きかえて、成績と計算量を比べてみましょう。
