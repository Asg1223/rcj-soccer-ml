# 3-3 過学習とデータの量

## ねらい

- **過学習**（学習用データにだけ強くなること）が起きているかを、損失のグラフで見分けられる
- 早期終了（EarlyStopping）、正則化（L2、ドロップアウト）の役割がわかる
- **データの量** で成績がどれだけ変わるかを、実験で確かめられる

---

## 理論 1：過学習とは

学習を続けると、学習用データの損失はどんどん下がります。でも、あるところから先は、**学習用データの細かいくせ（ノイズ）まで覚えはじめ**、初めて見るデータでは逆に悪くなることがあります。これを **過学習** といいます。

見分け方は、学習用と検証用の損失を並べて描くことです。

- 両方とも下がっている → まだ学習が足りない（もっと続けてよい）
- 学習用だけ下がり、検証用が **止まる・上がる** → 過学習が始まっている

## 理論 2：過学習を防ぐ方法

| 方法 | やっていること |
|---|---|
| **データを増やす** | 覚えきれないほど多くの例を見せる。いちばん効く |
| **早期終了** | 検証用の損失が良くならなくなったら止める |
| **L2 正則化** | 重みが大きくなりすぎないように、損失に「重みの大きさ」を足す |
| **ドロップアウト** | 学習中、ランダムにニューロンを休ませる。特定のニューロンに頼りすぎないようにする |
| **ネットを小さくする** | 覚えられる量そのものを減らす |

---

## 手を動かす 1：わざと過学習させる

学習用データを **取り直し 1 回分（500 行）** だけにして、大きめのネット（128 → 128）を、早期終了なしで 500 エポック学習させます。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from rcjml import load_config
from rcjml.data import load_synthetic, split_by_session
from rcjml.nn import features, targets, evaluate, keras_loss, build_ballnet

cfg = load_config()
df = load_synthetic("ring12_large_normal")
train_all, test = split_by_session(df)
val = train_all[train_all.session == 8]
one = train_all[train_all.session == 0]          # 500 行だけ

keras.utils.set_random_seed(0)
big = build_ballnet(hidden=(128, 128))
big.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras_loss())
h = big.fit(features(one, cfg), targets(one),
            validation_data=(features(val, cfg), targets(val)),
            epochs=500, batch_size=32, verbose=0)

plt.plot(h.history["loss"], label="train")
plt.plot(h.history["val_loss"], label="validation")
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
best = int(np.argmin(h.history["val_loss"]))
print(f"学習用の損失（最後）: {h.history['loss'][-1]:.3f}")
print(f"検証用の損失（いちばん良いとき）: {min(h.history['val_loss']):.3f}（{best} エポック目）")
print(f"検証用の損失（最後）: {h.history['val_loss'][-1]:.3f}")
```

学習用の損失は 0.03 くらいまで下がりますが、検証用は 0.08 くらいで止まり、最後は少し上がります。**学習用と検証用の差が大きい** のが、過学習のしるしです。

## 手を動かす 2：早期終了と正則化

```python
def fit_eval(train, val, test, hidden=(64, 64), epochs=400, l2=0.0, dropout=0.0,
             early_stop=True, batch_size=64, seed=0):
    keras.utils.set_random_seed(seed)
    model = build_ballnet(hidden=hidden, l2=l2, dropout=dropout)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras_loss())
    cb = [keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)] if early_stop else []
    h = model.fit(features(train, cfg), targets(train),
                  validation_data=(features(val, cfg), targets(val)),
                  epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=0)
    r = evaluate(model.predict(features(test, cfg), verbose=0), test)
    r["epochs"] = len(h.history["loss"])
    return r

print("早期終了なし :", round(evaluate(big.predict(features(test, cfg), verbose=0), test)["angle_MAE"], 1))
r1 = fit_eval(one, val, test, hidden=(128, 128), epochs=500, batch_size=32)
r2 = fit_eval(one, val, test, hidden=(128, 128), epochs=500, batch_size=32, l2=1e-4, dropout=0.1)
print("早期終了あり :", round(r1["angle_MAE"], 1), f"（{r1['epochs']} エポックで停止）")
print("＋L2＋ドロップアウト:", round(r2["angle_MAE"], 1), f"（{r2['epochs']} エポックで停止）")
```

| 方法（取り直し 1 回分で学習） | 角度の平均誤差 |
|---|---|
| 早期終了なし（500 エポック） | 7.4° |
| 早期終了あり | 7.5° |
| 早期終了 ＋ L2 ＋ ドロップアウト | 8.5° |

このデータでは、過学習による悪化は小さく、早期終了は「成績はほぼ同じで、無駄な学習を省く」役割になりました。正則化は、かけすぎると逆に悪くなることもあります。**どの方法も「効くかどうかは検証用データで確かめる」** のが基本です。

## 手を動かす 3：データの量を変える

同じネット（64 → 64、早期終了あり）で、学習用の取り直しの数だけを変えます。最後の行は、シミュレータで同じ機体のデータを 22,000 行作って学習させたものです。テスト用データはどれも同じです。

```python
from rcjml.sim import make_dataset

rows = {}
for k in [1, 2, 4, 8]:
    sub = train_all[train_all.session < k]
    rows[f"{k} sessions ({len(sub)} rows)"] = fit_eval(sub, val, test)["angle_MAE"]

big_data = make_dataset(cfg, n=24000, difficulty="normal", sessions=48, seed=7)
sim_train = big_data[big_data.session < 44]
sim_val = big_data[(big_data.session >= 44) & (big_data.session < 46)]
rows[f"simulator ({len(sim_train)} rows)"] = fit_eval(sim_train, sim_val, test)["angle_MAE"]
pd.Series(rows).round(1)
```

| 学習用データ | 角度の平均誤差 |
|---|---|
| 取り直し 1 回分（500 行） | 8.8° |
| 取り直し 2 回分（1,000 行） | 9.4° |
| 取り直し 4 回分（2,000 行） | 7.4° |
| 取り直し 8 回分（4,000 行） | 5.8° |
| シミュレータで作ったデータ（22,000 行） | 5.2° |

1 回分と 2 回分の差は、学習のたびに変わるランダムさ（重みの最初の値など）と同じくらいの大きさで、はっきりした差ではありません（`seed` を変えて何回かやると、平均ではほぼ同じになります）。でも、全体としては **データが多いほど良くなります**。ニューラルネットの成績を上げるいちばん確かな方法は、ネットを工夫することより、**良いデータをたくさん集めること** です。実機なら、1-4 の取り直しを、条件を変えながら何回もやることにあたります。

!!! example "やってみよう"
    1. `hard` のデータで、データの量を変える実験をしてみましょう。`make_dataset(..., difficulty="hard")` で 24,000 行作ると、どこまで下がるでしょう。
    2. ネットを `hidden=(16,)` のように小さくすると、データが少ないときと多いときで、成績はどう変わるでしょう。

---

## チェック問題

1. 学習用の損失は下がり続けているのに、検証用の損失が上がり始めました。何が起きていますか。どうしますか。
2. 「テスト用データの成績が良くなるまで、何度もネットを作り直す」のは、何がいけないのでしょう。
3. データを増やすのが一番効くとわかっても、実機でたくさん集めるのは大変です。どんな工夫がありますか。

??? note "答え"
    1. 過学習です。早期終了で止める、データを増やす、正則化を入れる、ネットを小さくする、などを試します。
    2. テスト用データに合わせてネットを選んでいるので、テスト用データを間接的に学習してしまい、本番での実力より良く見えます（2-4）。調整は検証用データで行います。
    3. 1-4 の `plan` で手順を決めて手早く集める、回転台で自動化する、データ拡張（3-5）、シミュレータで学んでから実機の少しのデータで仕上げる（7-1）などです。

## つまずきポイント

??? failure "早期終了で、すぐ（数エポックで）止まってしまう"
    `patience`（何エポック待つか）が小さすぎるか、学習率が大きすぎて検証用の損失がばたついています。`patience=20〜50` にしてみます。

## もっと深く

!!! deeper "バイアスとバリアンス"
    モデルの誤差は、おおまかに

    - **バイアス**：モデルが単純すぎて、そもそも表せない誤差（線形回帰で場合分けができない、など）
    - **バリアンス**：データが少なくて、たまたまのくせを覚えてしまう誤差（過学習）

    に分けて考えられます。ネットを大きくするとバイアスは減りますがバリアンスは増え、データを増やすとバリアンスが減ります。「学習用の成績も悪い」ならバイアス、「学習用は良いのに検証用が悪い」ならバリアンスが主な原因です。どちらかによって、打つ手が変わります。
