# 1-5 データを眺める

## ねらい

- 集めたデータを表とグラフで確かめ、おかしなところに気づける
- 「どのセンサが、どの角度のボールに反応しているか」を1枚の図で見られる
- 従来の計算方法が大きく外す場面を探し、その理由を考えられる

機械学習の前に、**まずデータを自分の目で見る** のが鉄則です。ここで気づいたことが、第2部以降のモデルづくりのヒントになります。

---

## 理論：何を見ればよいか

| 確かめること | 見方 |
|---|---|
| 数はそろっているか | 取り直しごとの行数 |
| 値の範囲はおかしくないか | 最小・最大・平均、0 や最大値に張りついている割合 |
| ボールの方向にちゃんと反応しているか | 角度ごとに平均した「反応の地図」（ヒートマップ） |
| 距離で値がどう変わるか | 正面のセンサの値と距離の関係 |
| 取り直しで条件が変わっているか | 取り直しごとの明るさの違い |
| 変なデータはないか | 従来の方法で大きく外れるデータ |

---

## 手を動かす

=== "Colab / ローカル"

    `notebooks/01_ir_data.ipynb` の後半に、このページのコードが入っています。

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asg1223/rcj-soccer-ml/blob/main/notebooks/01_ir_data.ipynb)

### 1. 読み込む

教材の合成データ（`normal`）を読み込みます。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rcjml import load_config
from rcjml.data import load_synthetic, sensor_columns

cfg = load_config()
df = load_synthetic("ring12_large_normal")
S = sensor_columns(df)          # ['s0', 's1', ..., 's11']

print(df.shape)                 # (行の数, 列の数)
df.head()
```

!!! private "自分のデータを使うとき"
    1-4 で集めた実機のデータを使うときは、`load_synthetic(...)` の代わりに次のようにします（`RCJML_CONFIG` でチーム用の設定を指定しておく）。

    ```python
    from rcjml.data import load_log
    df = load_log(cfg)
    ```

    実機のデータを読み込んだノートブックは、チーム用リポジトリかドライブに保存します。

### 2. 数と範囲を確かめる

```python
print(df.groupby("session").size())          # 取り直しごとの行数
print(df[S].describe().round(0))              # センサごとの平均・最小・最大など
print("0 の割合     :", (df[S] == 0).mean().mean().round(2))
print("最大値の割合 :", (df[S] == cfg.ir.max_reading).mean().mean().round(3))
```

合成データでは、取り直しごとに 500 行ずつあります。値の **14% が 0**（暗くて反応しない）、**3% が最大値**（近すぎて飽和）です。実機のデータで、0 や最大値の割合が極端に多いセンサがあったら、配線や取り付けを疑います。

### 3. いくつか描いてみる

ランダムに8行選んで、センサの値と本当のボールの方向を描きます。

```python
from rcjml.plot import polar_grid

sample = df.sample(8, random_state=0)
titles = [f"{a:+.0f} deg, {d:.0f} cm" for a, d in zip(sample.angle_deg, sample.distance_cm)]
polar_grid(cfg, sample[S].values, titles=titles, true_angles=sample.angle_deg.values)
plt.show()
```

### 4. 反応の地図（ヒートマップ）

ボールの角度を 15° ごとに区切り、それぞれでセンサの値を平均します。**横軸がボールの角度、縦軸がセンサ** の表を、色で描きます。

```python
d = df[df.has_ball == 0].copy()
d["angle_bin"] = (d["angle_deg"] // 15) * 15 + 7.5     # 15° ごとの区間の真ん中
table = d.groupby("angle_bin")[S].mean()

fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(table.T.values, aspect="auto", origin="lower",
               extent=[-180, 180, -0.5, len(S) - 0.5], cmap="viridis")
ax.set_xlabel("ball angle [deg]")
ax.set_ylabel("sensor")
ax.set_yticks(range(len(S)), S)
fig.colorbar(im, label="mean reading")
plt.show()
```

明るい帯が **ななめ** に走っていれば、「ボールの角度が変わると、反応するセンサも順番に変わる」ということです。

!!! example "やってみよう"
    1. 帯が途中で暗くなったり、太さが変わったりしているところはありますか。1-1 のどのくせが原因だと思いますか。
    2. `load_synthetic("ring12_large_easy")` に変えて、同じ図を描いて比べましょう。

??? note "やってみよう 1 のヒント"
    この合成データの機体では、`s2`・`s3`・`s9` の帯がほかより明るくなっています。この3つのセンサの **感度が高い**（個体差）からです。帯の太さがセンサごとに違うのは、**見える範囲の広さ**（指向性）の個体差です。

### 5. 距離で値がどう変わるか

正面（±15°）にボールがあるときの、正面のセンサ `s0` の値と距離の関係を描きます。取り直しごとに色を変えます。

```python
front = df[(df.angle_deg.abs() < 15) & (df.has_ball == 0)]

fig, ax = plt.subplots(figsize=(6, 4))
for sid, g in front.groupby("session"):
    ax.scatter(g.distance_cm, g.s0, s=8, alpha=0.6, label=f"session {sid}")
ax.set_xscale("log")
ax.set_xlabel("distance [cm] (log scale)")
ax.set_ylabel("s0")
ax.legend(fontsize=7, ncol=2)
plt.show()
```

横軸を **対数目盛り** にすると、点がだいたい右下がりの直線に並びます（1-1 の「もっと深く」の話です）。ただし、**同じ距離でも縦にかなりばらつく** ことがわかります。これでは、値1つから距離を正確に当てるのは難しそうです。

### 6. 取り直しで条件が変わっているか

ボールが 30〜60 cm にあるとき、いちばん大きいセンサの値を、取り直しごとに平均します。

```python
mid = df[df.distance_cm.between(30, 60) & (df.has_ball == 0)]
peak = mid[S].max(axis=1)
print(peak.groupby(mid["session"]).mean().round(0))
```

合成データでは、取り直しによって **618〜712** と、平均で 1 割以上違います。ボールの明るさや外乱光が、取り直しごとに変わっているからです。第2部で、この違いが「テストのやり方」に関わってきます。

### 7. 従来の方法が大きく外すデータを探す

第2部で使う従来の方法（重み付きベクトル和）で角度を計算し、**45° 以上外れた** データを探します。

```python
from rcjml.sim import baseline_angle, angle_error_deg

df["base_err"] = angle_error_deg(baseline_angle(cfg, df[S].values), df["angle_deg"])
odd = df[df.base_err > 45]
print(f"大きく外したデータ: {len(odd)} / {len(df)} 行")
print("その距離:", odd.distance_cm.describe().round(0).to_dict())

# 合成データだけにある「答え合わせ用の印」で、原因を確かめる
flags = ["reflection", "occluded", "stray"]
print(pd.DataFrame({"全体": df[flags].mean(), "大きく外したデータ": odd[flags].mean()}).round(2))
```

合成データでは、大きく外したのは全体の **1.7%** です。その多くは **遠い**（真ん中の値で 151 cm）データでした。印を見ると、相手ロボットによる遮へい（`occluded`）は全体の 12% なのに、大きく外したデータでは 47% あります。ボール以外の赤外線（`stray`）も、全体の 5% に対して 41% です。**暗いボールの光が、ほかの光に負けている** ことがわかります。

```python
worst = odd.sort_values("base_err", ascending=False).head(8)
titles = [f"err {e:.0f} deg" for e in worst.base_err]
polar_grid(cfg, worst[S].values, titles=titles, true_angles=worst.angle_deg.values,
           pred_angles=baseline_angle(cfg, worst[S].values))
plt.show()
```

緑の点線が従来の方法の答え、オレンジが本当の方向です。

!!! private "実機のデータには印がない"
    `reflection` などの印は、シミュレータが「答え合わせ用」に付けているだけです。実機のデータにはありません。実機では、大きく外したデータをグラフで1つずつ見て、「後ろに山がもう1つある → 反射かも」のように推理します。

---

## チェック問題

1. ヒートマップの明るい帯が、ななめではなく **水平** だったら、何が起きていると考えられますか。
2. 距離のグラフで、同じ距離なのに値が縦にばらつく理由を、1-1 のくせから2つ挙げましょう。
3. 大きく外したデータが「遠いボール」に多いのはなぜでしょう。

??? note "答え"
    1. ボールの角度に関係なく、同じセンサがいつも反応しています。外乱光がそのセンサにずっと当たっているか、配線の不良などが考えられます。
    2. たとえば、取り直しごとのボールの明るさの違い、読み取りのばらつき、外乱光、反射、遮へいなど。
    3. 遠いとボールの光が暗くなり、反射やボール以外の赤外線、外乱光の影響が相対的に大きくなるからです。

## つまずきポイント

??? failure "`KeyError: 'session'` と出る"
    自分で作った CSV に `session` の列がありません。1-3・1-4 のロガーで集めたデータには入っています。別の方法で集めたなら、列を足してください。

??? failure "ヒートマップが真っ暗（または真っ白）"
    ほとんどの値が 0（または最大値）です。センサの読み取り（1-2）を確かめます。

## もっと深く

!!! deeper "データを見るときに大事なこと"
    - **平均だけで判断しない**：平均がよくても、一部のデータで大きく外していることがあります。この章でやったように、**ひどいものから順に** 中身を見るのが近道です
    - **見てから決める**：「遮へいされたデータは捨てる」と決める前に、本番でもそれが起きるかを考えます。本番で起きるなら、モデルが慣れておくべきデータです
    - **答えを見すぎない**：テスト用のデータ（2-4）を何度も眺めて、それに合わせてモデルを工夫すると、テストの点数が実力以上に良く見えます。データを眺めるのは、学習用のデータだけにするのが理想です（この章では、わかりやすさのために全部を見ました）
