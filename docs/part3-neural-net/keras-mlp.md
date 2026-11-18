# 3-2 Keras で MLP：角度・距離・保持をまとめて推定する

## ねらい

- Keras でニューラルネット（MLP）を作り、学習させられる
- 1つのネットで、**角度・距離・ボールを保持しているか** の3つを同時に推定できる
- 第2部の方法と成績を比べられる

---

## 理論 1：1つのネットで3つの答えを出す

この教材のネット（**BallNet** と呼びます）は、出力を4つ持たせます。

| 出力 | 中身 | 理由 |
|---|---|---|
| `out[0]`, `out[1]` | 角度の cos と sin | 角度は輪っかだから（2-3） |
| `out[2]` | 距離の **log** を、だいたい -2〜2 にそろえたもの | 距離は 3〜180 cm と範囲が広いので、log にして大きさをそろえる |
| `out[3]` | 保持しているかの「ロジット」 | シグモイドに通すと 0〜1 の確率になる |

3つを別々のネットにしてもよいのですが、まとめると **隠れ層を共有できる** ので、全体が小さくなります。マイコンに載せるときに有利です。

## 理論 2：損失を組み合わせる

答えの種類によって、ちょうどよい損失が違います。

- 角度と距離（数の大きさを当てる **回帰**）：**二乗誤差**（2-2 と同じ）
- 保持（はい・いいえを当てる **分類**）：**交差エントロピー**（3-4 でくわしく）

BallNet では、この2つを足したものを損失にします。教材の `rcjml.nn.keras_loss()` がそれです。

## 理論 3：学習のやり方

2-2 と同じ勾配降下法ですが、ニューラルネットでは次の工夫をします。

- **ミニバッチ**：全データではなく、64 行ずつ取り出して重みを直す（速く、ほどよくばらつく）
- **Adam**：坂の傾きに合わせて、学習率を自動で調整してくれる方法
- **エポック**：学習用データを1周することを 1 エポックという
- **検証用データ**：学習用とは別の取り直しを1つ取っておき、学習の途中で成績を見る（2-4 の「もっと深く」）

---

## 手を動かす 1：データを用意する

=== "Colab / ローカル"

    `notebooks/03_neural_net.ipynb` に、第3部のコードが全部入っています。Colab なら「ランタイム → ランタイムのタイプを変更」で GPU にすると少し速くなりますが、このネットは小さいので CPU でも 1 分ほどで学習できます。

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asg1223/rcj-soccer-ml/blob/main/notebooks/03_neural_net.ipynb)

    ローカルで動かすときは、先に `pip install tensorflow` を実行します（Python 3.12 推奨、0-2）。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from rcjml import load_config
from rcjml.data import load_synthetic, split_by_session
from rcjml.nn import features, targets, decode, evaluate, keras_loss

cfg = load_config()
df = load_synthetic("ring12_large_normal")
train_all, test = split_by_session(df)            # テスト用：最後の 1/4 の取り直し

last = train_all["session"].max()
train = train_all[train_all["session"] < last]     # 学習用
val = train_all[train_all["session"] == last]      # 検証用：学習用の最後の取り直し

X_train, Y_train = features(train, cfg), targets(train)
X_val, Y_val = features(val, cfg), targets(val)
X_test = features(test, cfg)
print(X_train.shape, Y_train.shape)               # (4000, 12) (4000, 4)
print(Y_train[:3].round(2))                        # cos, sin, 距離, 保持
```

## 手を動かす 2：ネットを作る

```python
keras.utils.set_random_seed(0)      # 毎回同じ結果になるようにする

model = keras.Sequential([
    keras.Input(shape=(12,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(4),                    # 出力：活性化関数なし
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=keras_loss())
model.summary()
```

`Total params: 5,252` と出れば、3-1 で数えたのと同じ大きさのネットです。

## 手を動かす 3：学習させる

```python
stop = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=400, batch_size=64,
    callbacks=[stop], verbose=0,
)
print("学習したエポック数:", len(history.history["loss"]))

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.yscale("log")
plt.legend()
plt.show()
```

`EarlyStopping` は「検証用の損失が 20 エポック続けて良くならなかったら止めて、いちばん良かったときの重みに戻す」という仕組みです（3-3 でくわしく）。

## 手を動かす 4：テスト用データで成績を見る

```python
P = model.predict(X_test, verbose=0)
result = evaluate(P, test)
for k, v in result.items():
    print(f"{k:18s} {v:.3f}")
```

| 項目 | 結果（normal） |
|---|---|
| 角度の平均誤差 | 5.8° |
| 距離の誤差（割合の真ん中の値） | 15% |
| 保持の正解率 | 99.5% |

角度は、第2部の方法と並べるとこうなります。

| 方法 | normal | hard | ゴルフボール（normal） | 大きさ |
|---|---|---|---|---|
| 重み付きベクトル和 | 9.4° | 19.8° | 17.9° | 重み 24 個 |
| ランダムフォレスト（2-5） | 4.3° | 16.8° | — | 約 90 MB |
| **BallNet** | **5.8°** | **18.6°** | **16.1°** | **約 21 KB** |

ランダムフォレストにはまだ少し負けていますが、**大きさは 4,000 分の 1 以下** で、距離と保持も同時に出せます。そして、データを増やすとまだ良くなります（3-3）。

## 手を動かす 5：距離の推定を見る

```python
d = decode(P)
plt.figure(figsize=(5, 5))
plt.scatter(test["distance_cm"], d["distance_cm"], s=4, alpha=0.4)
plt.plot([3, 180], [3, 180], color="tab:orange")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("true distance [cm]")
plt.ylabel("predicted distance [cm]")
plt.show()
```

点がオレンジの線（予測 ＝ 正解）の近くに集まっていれば、うまく推定できています。1-5 で見た「同じ距離でも値がばらつく」データから、ここまで推定できます。

!!! example "やってみよう"
    1. 隠れ層を `Dense(16)` 1 層だけにすると、成績と大きさはどう変わりますか。
    2. `load_synthetic("ring12_large_hard")` に変えて、同じことをしてみましょう。
    3. 学習率を `1e-2` や `1e-4` にすると、学習の進み方はどう変わりますか。

---

## チェック問題

1. 距離をそのまま（cm で）出力せず、log にしてそろえるのはなぜですか。
2. 検証用データとテスト用データは、何がちがいますか。
3. BallNet の出力 `out[3]` を、保持している確率に直すにはどうしますか。

??? note "答え"
    1. 3 cm と 180 cm では値の大きさが 60 倍も違い、二乗誤差だと遠いボールの誤差ばかりが大きく効いてしまうからです。log にすると「何割ずれたか」で比べることになり、近くも遠くも同じくらい大事にできます。
    2. 検証用は「学習の途中で、止めどきや設定を決めるために見る」データ、テスト用は「最後に1回だけ、成績をつけるために使う」データです。
    3. シグモイド関数 $1 / (1 + e^{-z})$ に通します（`decode` の中でやっています）。

## つまずきポイント

??? failure "`ModuleNotFoundError: No module named 'keras'` / `tensorflow`"
    ローカルなら `pip install tensorflow`（keras も一緒に入る）。Python 3.14 だと入らないことがあるので、3.12 の仮想環境を使います（0-2）。

??? failure "実行するたびに結果が少し変わる"
    重みの最初の値やミニバッチの順番がランダムだからです。`keras.utils.set_random_seed(0)` で固定できます。比べるときは、seed を変えて数回やって平均を見ると確かです。

??? failure "損失が `nan` になる"
    入力を 0〜1 にそろえ忘れている（`features` を使わずに生の値を入れている）か、学習率が大きすぎます。

## もっと深く

!!! deeper "損失の重みづけ"
    BallNet の損失は「位置の二乗誤差 ＋ 保持の交差エントロピー」です。どちらをどれくらい重視するかは `keras_loss(hold_weight=...)` で変えられます。保持の判定がもっと大事なら重みを上げます。
    複数の答えを1つのネットで同時に学ぶことを **マルチタスク学習** といいます。お互いに役立つ情報（「ボールが近い」ことは、距離にも保持にも関係する）を隠れ層で共有できるので、別々に学ぶより良くなることがあります。逆に、相性の悪い答えを一緒にすると、どちらも悪くなることもあります。
