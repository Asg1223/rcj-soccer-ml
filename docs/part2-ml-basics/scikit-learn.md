# 2-5 scikit-learn で数行に

## ねらい

- scikit-learn の「`fit` で学習、`predict` で予測」という使い方がわかる
- 2-2 で一から書いた線形回帰と、同じ結果になることを確かめる
- 線形ではないモデル（k 近傍法、ランダムフォレスト）を試し、従来の方法と比べる

---

## 理論 1：scikit-learn の使い方は全部同じ

**scikit-learn** は、機械学習のいろいろなモデルを同じ使い方で試せるライブラリです。どのモデルも、

```text
model = モデルの名前(設定)
model.fit(X_train, Y_train)      # 学習
P = model.predict(X_test)        # 予測
```

の3行で使えます。モデルを取りかえても、残りのコードはそのままです。

## 理論 2：線形ではないモデル

2-2 で見たとおり、線形モデルは「かけて足す」ことしかできません。そこで、**場合分け** ができるモデルを2つ試します。

| モデル | 考え方 | たとえ |
|---|---|---|
| **k 近傍法**（KNN） | 学習用データの中から、値がいちばん **似ている k 個** を探し、その正解の平均を答える | 「前にも同じような値を見たことがある。そのときボールはここだった」 |
| **ランダムフォレスト** | 「s3 が 400 より大きい？」のような質問を重ねた **決定木** をたくさん作り、答えを平均する | 質問ゲームの名人を何百人も集めて、多数決をとる |

---

## 手を動かす 1：線形回帰を scikit-learn で

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from rcjml import load_config
from rcjml.data import load_synthetic, sensor_columns, split_by_session
from rcjml.sim import angle_error_deg

cfg = load_config()
df = load_synthetic("ring12_large_normal")
S = sensor_columns(df)
train, test = split_by_session(df)

def xy(d):
    th = np.deg2rad(d["angle_deg"].values)
    return d[S].values / cfg.ir.max_reading, np.c_[np.cos(th), np.sin(th)]

def to_angle(P):
    return np.rad2deg(np.arctan2(P[:, 1], P[:, 0]))

X_train, Y_train = xy(train)
X_test, _ = xy(test)

model = LinearRegression()
model.fit(X_train, Y_train)
err = angle_error_deg(to_angle(model.predict(X_test)), test["angle_deg"])
print(f"LinearRegression: {err.mean():.1f}°")
```

2-2 で一から書いた線形回帰と同じ **8.9°** になります。中でやっていることは同じです（scikit-learn は、勾配降下法ではなく最小二乗法で一発で解いています）。

## 手を動かす 2：モデルを取りかえて比べる

```python
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from rcjml.sim import baseline_angle

models = {
    "linear": LinearRegression(),
    "KNN (k=5)": KNeighborsRegressor(n_neighbors=5),
    "random forest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0),
}

results = {"vector sum (2-1)": angle_error_deg(baseline_angle(cfg, test[S].values), test["angle_deg"]).mean()}
for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, Y_train)
    results[name] = angle_error_deg(to_angle(model.predict(X_test)), test["angle_deg"]).mean()
    print(f"{name:14s} 学習 {time.time() - t0:.1f} 秒")

pd.Series(results).round(1)
```

| 方法 | normal | hard |
|---|---|---|
| 重み付きベクトル和（2-1） | 9.4° | 19.8° |
| 線形回帰 | 8.9° | 20.3° |
| k 近傍法（k=5） | 5.0° | 17.3° |
| ランダムフォレスト | **4.3°** | **16.8°** |

（hard の列は、`load_synthetic("ring12_large_hard")` に変えて同じことをした結果）

`normal` では、場合分けのできるモデルで **誤差が半分以下** になりました。`hard` でも従来の方法より良くなりますが、まだ大きな誤差が残っています。

## 手を動かす 3：どこが良くなったか見る

従来の方法とランダムフォレストの誤差を、距離ごとに比べます。

```python
import matplotlib.pyplot as plt

rf = models["random forest"]
err_rf = angle_error_deg(to_angle(rf.predict(X_test)), test["angle_deg"])
err_base = angle_error_deg(baseline_angle(cfg, test[S].values), test["angle_deg"])

bins = pd.cut(test["distance_cm"], [0, 20, 40, 80, 120, 180])
table = pd.DataFrame({"vector sum": err_base, "random forest": err_rf}).groupby(bins.values, observed=True).mean()
table.round(1).plot.bar(figsize=(7, 3.5), rot=0, ylabel="mean angle error [deg]")
plt.show()
```

どの距離でも良くなっていて、特に **遠いボール** での改善が大きいことがわかります。

## でも、このモデルはロボットに載るか？

成績だけ見るとランダムフォレストが一番ですが、ロボットで使うには別の問題があります。

```python
import pickle
for name, model in models.items():
    print(f"{name:14s} 大きさ {len(pickle.dumps(model)) / 1e6:8.2f} MB")
```

| モデル | 大きさ（normal で学習） |
|---|---|
| 線形回帰 | 0.00 MB（重み 26 個） |
| k 近傍法 | 0.60 MB（学習用データを全部持っている） |
| ランダムフォレスト | 約 90 MB |

Arduino Uno のメモリは 2 KB、ESP32-S3 でも数 MB です。ランダムフォレストはそのままでは **とても載りません**。k 近傍法も、毎回すべての学習用データと比べるので、マイコンでは遅すぎます。

そこで第3部では、**小さくて、場合分けもできる** モデルとして **ニューラルネット** を使います。そして第4部で、それをマイコンに載せます。

!!! example "やってみよう"
    1. k 近傍法の `n_neighbors` を 1, 5, 20, 50 と変えると、誤差はどう変わるでしょう。
    2. ランダムフォレストの `n_estimators`（木の数）を 10 にすると、誤差と大きさはどう変わるでしょう。
    3. ゴルフボールサイズのデータ（`ring12_golf_normal`）で比べてみましょう。

---

## チェック問題

1. scikit-learn でモデルを取りかえるとき、コードのどこを変えればよいですか。
2. k 近傍法は、学習（`fit`）がほとんど一瞬で終わります。なぜでしょう。
3. 成績がいちばん良いモデルを、そのままロボットに使えないことがあるのはなぜですか。

??? note "答え"
    1. `model = ...` の行だけです。`fit` と `predict` はどのモデルでも同じです。
    2. k 近傍法の「学習」は、学習用データを覚えておくだけだからです。そのぶん、予測のたびに全部のデータと比べるので、予測は遅くなります。
    3. モデルが大きすぎてマイコンのメモリに入らない、予測に時間がかかりすぎてロボットの動きに間に合わない、などの理由です。

## つまずきポイント

??? failure "`ModuleNotFoundError: No module named 'sklearn'`"
    scikit-learn が入っていません。ローカルなら `pip install scikit-learn`（パッケージ名は `scikit-learn`、読み込むときの名前は `sklearn`）。Colab には最初から入っています。

??? failure "ランダムフォレストの学習が遅い"
    `n_jobs=-1` で、PC のすべての CPU を使うようにします。それでも遅ければ `n_estimators` を減らします。

## もっと深く

!!! deeper "特徴量を工夫する"
    モデルを変える以外に、**入力の形を工夫する**（特徴量エンジニアリング）方法もあります。たとえば、

    - 全センサの合計で割る：ボールの明るさの違い（取り直しごとの違い）を打ち消せる
    - いちばん大きいセンサの番号と、その両どなりの値：どこに山があるかを直接教える
    - 重み付きベクトル和の結果（cos, sin）を入力に足す：従来の方法の答えを「ヒント」として渡す

    線形回帰でも、良い特徴量を足すとずっと良くなることがあります。ただし、特徴量の工夫は **学習用データだけ** を見て考えます（2-4）。どの工夫が効くか、`hard` のデータで試してみましょう。
