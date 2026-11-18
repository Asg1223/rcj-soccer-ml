# 7-1 会場での調整：少ないデータで追加学習

## ねらい

- 家（練習場）で学習したモデルが、会場では大きくずれることを確かめる
- 会場で集めた **少しのデータ** で、モデルを調整（**追加学習**、fine-tuning）できる
- 「ゼロから学習」「追加学習」「混ぜて追加学習」の違いと、使い分けがわかる

---

## 理論 1：会場は「知らないセッション」

第2部で、データは **取り直し（セッション）で分けて評価する** と学びました。取り直すと、外乱光やボールの明るさが少し変わるからです。

大会の会場は、その **いちばん大きな取り直し** です。

- 照明が違う（窓からの日光、強いライト）。ある方向からだけ強い外乱光が入ることもある
- ボールの電池の減り方が違い、明るさが違う
- 床やかべの反射が違う

しかも、会場でデータを集められる時間は短いです（試合の合間の数分〜数十分）。

## 理論 2：追加学習

ゼロから学習し直す代わりに、**家で学習した重みから始めて**、会場のデータで少しだけ学習を続けます。

- 家のモデルは、「12 個のセンサの値からボールの方向を出す」大まかな仕組みをもう知っている
- 会場のデータは、そのずれを直す分だけあればよい
- 学習率を小さくして（例：1/3）、少ない回数だけ学習する

家のデータを少し **混ぜて** 追加学習すると、会場に合わせすぎて家（ほかの会場）のことを忘れるのを防げます。

---

## 手を動かす 1：家のモデルと「会場」のデータ

```python
import numpy as np
import pandas as pd
import keras
from rcjml import load_config
from rcjml.data import load_synthetic, split_by_session
from rcjml.nn import features, targets, keras_loss, build_ballnet, evaluate
from rcjml.sim import make_dataset, Session

cfg = load_config()
df = load_synthetic("ring12_large_normal")
train, test = split_by_session(df)
last = train.session.max()
home_fit, home_val = train[train.session < last], train[train.session == last]

def fit(model, trn, val, epochs=400, lr=1e-3, patience=20):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=keras_loss())
    model.fit(features(trn, cfg), targets(trn), validation_data=(features(val, cfg), targets(val)),
              epochs=epochs, batch_size=64, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)])
    return model

keras.utils.set_random_seed(0)
home = fit(build_ballnet(), home_fit, home_val)

# 会場：左前（70°）の方向から強い光が入り、ボールが暗め
venue_condition = Session(ambient=1.5, ambient_dir_deg=70.0, ambient_dir=5.0, ball_scale=0.75)
venue = make_dataset(cfg, n=3000, sessions=6, seed=77, venue=venue_condition)
venue_cal = venue[venue.session == 0]      # 会場で最初に集めるデータ（ここから少しだけ使う）
venue_test = venue[venue.session >= 1]     # その後の試合のつもり

mae = lambda m, d: evaluate(m.predict(features(d, cfg), verbose=0), d)["angle_MAE"]
print(f"家のモデル：家で {mae(home, test):.1f}°、会場で {mae(home, venue_test):.1f}°")
```

家では 5.8° だったのが、会場では **18.8°** までずれます。

## 手を動かす 2：3 つの方法で調整する

```python
def calibrate(n):
    cal = venue_cal.sample(n, random_state=0)
    c_fit, c_val = cal.iloc[: int(n * 0.8)], cal.iloc[int(n * 0.8):]

    keras.utils.set_random_seed(0)                    # (1) ゼロから
    scratch = fit(build_ballnet(), c_fit, c_val)

    keras.utils.set_random_seed(0)                    # (2) 家の重みから追加学習
    ft = build_ballnet()
    ft.set_weights(home.get_weights())
    ft = fit(ft, c_fit, c_val, epochs=200, lr=3e-4, patience=10)

    keras.utils.set_random_seed(0)                    # (3) 家のデータも混ぜて追加学習
    mix = build_ballnet()
    mix.set_weights(home.get_weights())
    mixed = pd.concat([home_fit.sample(2000, random_state=0), pd.concat([c_fit] * 5)])
    mix = fit(mix, mixed, c_val, epochs=200, lr=3e-4, patience=10)
    return scratch, ft, mix

for n in [50, 200, 500]:
    scratch, ft, mix = calibrate(n)
    print(f"会場のデータ {n:3d} 個: ゼロから {mae(scratch, venue_test):.1f}°  "
          f"追加学習 {mae(ft, venue_test):.1f}°（家で {mae(ft, test):.1f}°）  "
          f"混ぜて追加学習 {mae(mix, venue_test):.1f}°（家で {mae(mix, test):.1f}°）")
```

会場のテストデータでの角度の平均のずれです（かっこの中は、調整したあとのモデルを家のテストデータで測ったもの）。

| 会場のデータ | ゼロから | 追加学習 | 混ぜて追加学習 |
|---|---|---|---|
| 0 個（家のモデルのまま） | — | 18.8° | 18.8° |
| 50 個 | 10.9° | **7.8°**（家で 9.2°） | 9.3°（家で **5.7°**） |
| 200 個 | 8.2° | 6.7°（家で 9.3°） | **6.5°**（家で 6.0°） |
| 500 個 | 7.3° | **6.1°**（家で 10.0°） | 6.4°（家で 6.5°） |
| （参考）会場のデータ 4,000 個でゼロから | | | 5.7° |

ここからわかること：

1. **50 個だけでも大きく良くなる**（18.8° → 7.8°）。家のモデルから始めると、ゼロからより少ないデータですみます。
2. **追加学習だけだと、家のことを忘れる**（家で 5.8° → 9〜10°）。次の会場、別のコートでまたずれます。
3. **家のデータを混ぜると、両方そこそこ良い**。200 個あれば、会場でも家でもほぼ元どおりです。

!!! tip "会場での手順"
    1. 試合の前に、5-4 のデータ収集モードで **会場のコートで 200 個くらい** 集める（ボールをいろいろな角度・距離に置く。数分〜10 分くらい）。ボールなし（7-2）も少し集める。
    2. PC で「混ぜて追加学習」をして、第4部の方法でマイコン用に書き出す。
    3. **家のモデルも残しておく**。調整がうまくいかなかったら、すぐに戻せるようにする。
    4. 会場のデータも、調整したモデルも、チームの **非公開リポジトリ** に置く。

!!! private "会場で集めたデータとモデル"
    会場のデータ、調整したモデル、しきい値は、チームの大事な情報です。公開リポジトリには置きません（`data/` と `ballnet_model.h` は `.gitignore` で除外してあります）。

---

## チェック問題

1. 会場で、ゼロから学習するより追加学習のほうが少ないデータですむのはなぜですか。
2. 追加学習のとき、学習率を小さくするのはなぜですか。
3. 「混ぜて追加学習」で、会場のデータを 5 回くり返して混ぜたのはなぜですか。

??? note "答え"
    1. 家のモデルは、センサの値から方向を出す大まかな仕組みをもう知っているので、会場とのずれを直す分だけ学習すればよいからです。
    2. 大きく動かすと、家で覚えたことがこわれてしまうからです。少しずつ直します。
    3. 会場のデータ（数十〜数百個）が家のデータ（2,000 個）より少ないので、そのままだと会場のデータがほとんど効かないからです。

## つまずきポイント

??? failure "追加学習したら、かえって悪くなった"
    会場のデータが少なすぎるか、偏っています（同じ角度ばかり）。角度と距離がまんべんなく入るように集めます。また、評価用のデータは、調整に使ったのとは **別の時間** に集めたものを使います。

??? failure "会場の PC に Python の環境がない"
    家で環境を作ったノート PC を持っていきます。Colab はネットがないと使えないので、会場ではローカルの環境（第0部）を使えるようにしておきます。

## もっと深く

!!! deeper "転移学習とドメイン適応"
    ほかの場所（ドメイン）で学習したモデルを、新しい場所に合わせることを **ドメイン適応** と呼びます。このページの追加学習はそのいちばん簡単な方法です。
    - 最後の層だけ学習し直す、正規化（BatchNorm）の統計だけ会場のデータで計算し直す、といった **軽い調整** もあります。
    - 会場のデータにラベル（角度・距離）をつけずに合わせる方法（**教師なしドメイン適応**）も研究されています。
    - 忘れてしまう問題は **破滅的忘却**（catastrophic forgetting）と呼ばれ、継続学習の大きなテーマです。
