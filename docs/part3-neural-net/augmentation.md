# 3-5 データ拡張：センサリングを回して水増しする

## ねらい

- 手もとのデータから、新しい学習用データを作る **データ拡張** の考え方がわかる
- センサリングの **回転対称性** を使って、データを 12 倍に増やせる
- データ拡張が「効くとき」と「効かないとき」を、実験で確かめられる

---

## 理論 1：回しても同じ問題になる

センサが 30° おきに等間隔に並んでいるとします。ボールが 0° にあるときのセンサの値が

```text
s0=650, s1=420, s2=80, …, s11=430
```

だったとしたら、ロボットごと 30° 回した状況、つまり **ボールが 30° にあるとき** は、値が1つずつずれて

```text
s0=430, s1=650, s2=420, s3=80, …
```

になる「はず」です。つまり、1つのデータを **1つずつずらすたびに、角度を 30° 足した新しいデータ** が作れます。12 個のセンサなら、データを **12 倍** にできます。

これは、画像を左右反転させたり少し回したりして増やす、画像認識でよく使う方法と同じ考え方です。

## 理論 2：でも本当は「同じ」ではない

1-1 で見たとおり、実機には

- センサごとの **感度の個体差**
- センサごとの **取り付けのずれ**
- 機体の **柱やキャッチャーの影**（ロボットの前と後ろで形が違う）

があります。だから、ずらしたデータは「本物そっくりだけど、少しうそ」です。

- データが **少ない** とき：少しうそでも、たくさんの角度の例があるほうが助かる → **効く**
- データが **十分ある** とき：本物のデータで足りているので、うその分が邪魔になることもある → **あまり効かない**

!!! warning "保持しているデータは回さない"
    ボールを保持しているデータを回すと、「横でボールを保持している」というありえないデータになってしまいます。教材の `augment_rotations` は、保持しているデータは回さずにそのまま入れます。データ拡張では、**回したデータの正解が本当に正しいか** を必ず考えましょう。

---

## 手を動かす 1：回してみる

```python
import numpy as np
import pandas as pd
import keras
from rcjml import load_config
from rcjml.data import load_synthetic, split_by_session, sensor_columns
from rcjml.nn import features, targets, evaluate, keras_loss, build_ballnet, rotate_ring, augment_rotations

cfg = load_config()
df = load_synthetic("ring12_large_normal")
S = sensor_columns(df)

one = df[df.has_ball == 0].head(1)
print(one[S + ["angle_deg"]])
print(rotate_ring(one, 1)[S + ["angle_deg"]])       # 1 つずらすと、角度が +30°
```

## 手を動かす 2：学習用データが少ないときと多いときで比べる

3-3 と同じ `fit_eval` で、学習用データを **取り直し 3 回分**（1,500 行）にした場合と、**8 回分**（4,000 行）にした場合で、回して増やす前と後を比べます。

```python
train_all, test = split_by_session(df)
val = train_all[train_all.session == 8]

def fit_eval(train, val, test, epochs=400, seed=0):
    keras.utils.set_random_seed(seed)
    model = build_ballnet()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras_loss())
    stop = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    model.fit(features(train, cfg), targets(train),
              validation_data=(features(val, cfg), targets(val)),
              epochs=epochs, batch_size=64, callbacks=[stop], verbose=0)
    return evaluate(model.predict(features(test, cfg), verbose=0), test)

rows = {}
for name, k in [("3 sessions", 3), ("8 sessions", 8)]:
    sub = train_all[train_all.session < k]
    rows[(name, "original")] = fit_eval(sub, val, test)
    rows[(name, "rotated x12")] = fit_eval(augment_rotations(sub), val, test, epochs=200)
pd.DataFrame(rows).T[["angle_MAE", "dist_rel_median"]].round(3)
```

ノートブックでは数分かかります（回すとデータが 12 倍になるため）。

| 学習用データ | そのまま | 回して 12 倍 |
|---|---|---|
| 取り直し 3 回分（1,500 行） | 8.6° | **7.3°** |
| 取り直し 8 回分（4,000 行） | 5.8° | 5.7° |

データが少ないときは、はっきり良くなりました。データが多いときは、角度はほとんど変わりません（距離の推定は少し良くなります：15% → 14%）。

!!! example "やってみよう"
    1. `difficulty="easy"` のデータ（`ring12_large_easy`）で、同じ比べ方をしてみましょう。個体差が小さいと、回したデータは本物に近いはずです。効き方は変わるでしょうか。
    2. 12 通り全部ではなく、1 つずらし・11 個ずらし（±30°）だけ足すと、どうなるでしょう。

---

## チェック問題

1. センサが 8 個、等間隔のとき、1 つずらすと角度は何度変わりますか。
2. データ拡張で作ったデータは、テスト用データにも入れてよいでしょうか。
3. センサが等間隔に並んでいない機体では、この方法はそのまま使えますか。

??? note "答え"
    1. 45° です。
    2. いけません。テスト用データは「本番と同じ条件の本物のデータ」で成績をつけるためのものです。拡張は **学習用データだけ** に使います。
    3. そのままでは使えません。ずらした先のセンサの向きが、元のセンサの向きと違うからです。シミュレータで細かい角度ごとのデータを作る、近い角度どうしで補間する、などの工夫が要ります。

## つまずきポイント

??? failure "回したら成績が悪くなった"
    個体差や機体の影が大きいと、回したデータが「うそ」に近づきます。データが十分あるなら、無理に使わなくて構いません。データ拡張は「試して、検証用データで確かめて、効いたら使う」ものです。

## もっと深く

!!! deeper "シミュレータは究極のデータ拡張"
    この教材のシミュレータ自体が、「本物そっくりのデータを好きなだけ作る」道具です。3-3 で見たように、シミュレータでデータを増やすと成績は良くなります。
    ただし、シミュレータで学んだモデルを実機で使うと、シミュレータと実機の違い（**sim-to-real ギャップ**）のせいで、思ったほど良くならないことがよくあります。そこで、

    - シミュレータの条件をわざとばらばらに変えて学習させる（**ドメインランダム化**。この教材の取り直しごとの条件の変化がそれ）
    - シミュレータで学んだあと、実機の少しのデータで仕上げる（**ファインチューニング**、7-1）

    といった方法を組み合わせます。
