# H3 自分のデータで学習する

| 時間の目安 | 本文 | 使う機材 |
|---|---|---|
| 90 分 | [2-1](../part2-ml-basics/baseline.md)、[2-4](../part2-ml-basics/evaluation.md)、[3-2](../part3-neural-net/keras-mlp.md)、[3-3](../part3-neural-net/overfitting.md)、[3-4](../part3-neural-net/classification.md) | PC だけ |

## ゴール

- H2 で集めた **自分のデータ** で BallNet を学習する
- 従来の方法（重み付きベクトル和）と比べて、**取り直しで分けたテスト** で良くなったことを数字で示す
- 学習したモデルを、チーム用リポジトリに保存する

---

## ステップ

### 1. 学習して比べる（30 分）

チーム用リポジトリの `notebooks/` に新しいノートブックを作ります。

```python
import numpy as np
import keras
from rcjml import load_config
from rcjml.data import load_log, sensor_columns, split_by_session
from rcjml.nn import features, targets, keras_loss, build_ballnet, evaluate
from rcjml.sim import baseline_angle, angle_error_deg

cfg = load_config()
df = load_log(cfg, "ir_log.csv")
S = sensor_columns(df)
train, test = split_by_session(df, test_fraction=1 / 3)     # 3 回のうち最後の 1 回をテストに
print("学習:", sorted(train.session.unique().tolist()), len(train), "行 / テスト:", sorted(test.session.unique().tolist()), len(test), "行")

# (1) 従来の方法（第2部）
free = test.has_ball.values == 0
base = angle_error_deg(baseline_angle(cfg, test[S].values), test.angle_deg.values)[free].mean()
print(f"重み付きベクトル和: {base:.1f}°")

# (2) BallNet（第3部）
keras.utils.set_random_seed(0)
model = build_ballnet(n_in=len(S))
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras_loss())
model.fit(features(train, cfg), targets(train), validation_split=0.15, epochs=400, batch_size=64, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
r = evaluate(model.predict(features(test, cfg), verbose=0), test)
print({k: round(v, 3) for k, v in r.items()})
```

- [ ] 学習用とテスト用に、**別の取り直し** が入っていることを、最初の表示で確かめた
- [ ] 重み付きベクトル和と、BallNet の角度のずれ（`angle_MAE`）を記録シートに書いた
- [ ] 距離（`dist_rel_median`：何割ずれるか）と保持（`hold_recall`：保持を見のがさない割合、`hold_false_alarm`：保持していないのに保持と言う割合）も書いた

### 2. どこで外れるかを見る（20 分）

```python
import pandas as pd
from rcjml.nn import decode

d = decode(model.predict(features(test, cfg), verbose=0))
err = angle_error_deg(d["angle_deg"], test.angle_deg.values)
test = test.assign(err=err)
print(test[free].groupby((test.angle_deg // 45) * 45).err.mean().round(1))       # 角度ごと
print(test[free].groupby(pd.cut(test.distance_cm, [0, 20, 45, 80, 200])).err.mean().round(1))   # 距離ごと
```

- [ ] いちばん外れる角度と距離を、記録シートに書いた
- [ ] その角度の生の値を H2 の図で見て、理由を考えた（センサのすき間？反射？）

### 3. 改善を 1 つ試す（30 分）

次のうち **1 つ** を選んで試し、テストの数字が良くなるか確かめます。

- [ ] **データを増やす**：いちばん外れた角度・距離を、もう一度集める（学習用の取り直しとして）
- [ ] **データ拡張**：センサが等間隔なら、3-5 の `augment_rotations` を使う
- [ ] **モデルの大きさ**：`build_ballnet(n_in=len(S), hidden=(32, 32))` や `(128, 128)` と比べる（3-3）

!!! warning "テストのデータで選ばない"
    いくつも試して「テストでいちばん良かったもの」を選ぶと、テストの数字が実力より良く見えます（2-4）。本当はもう 1 回、取り直しを集めて、最後に 1 回だけ測るのが理想です。

### 4. 保存する（10 分）

```python
models_dir = cfg.source.parent.parent / "models"      # チーム用リポジトリの models/
models_dir.mkdir(exist_ok=True)
model.save(models_dir / "ballnet.keras")
print("保存しました:", models_dir / "ballnet.keras")
```

- [ ] チーム用リポジトリに保存して、コミットした（**教材のリポジトリではない** ことを確かめた）

---

## 合格条件

- [ ] 取り直しで分けたテストで、BallNet の角度のずれが **重み付きベクトル和より小さい**
- [ ] 保持の見のがし（1 − `hold_recall`）が 10% 以下（保持のデータが少ないときは、H2 に戻って足す）
- [ ] いちばん外れる角度・距離と、その理由の予想を記録した
- [ ] モデルがチーム用リポジトリに保存されている

## 記録シート

| 項目 | 重み付きベクトル和 | BallNet | 改善のあと |
|---|---|---|---|
| 角度のずれ（平均） | | | |
| 距離のずれ（中央値、割合） | — | | |
| 保持を見のがさない割合 | — | | |
| 保持の誤報 | — | | |
| いちばん外れる角度・距離 | | | |
| 試した改善 | | | |

---

## 困ったとき

??? failure "BallNet が重み付きベクトル和に負ける"
    データが少ないか、取り直しの間で条件が大きく違いすぎるかもしれません。まず取り直しを 1 回増やします。`hold` の行が少なすぎると、学習がそちらに引っぱられることもあります。

??? failure "`load_log` でファイルが見つからない"
    `robot.yaml` の `data.dir` と、ファイル名を確かめます。CoreS3 の SD から持ってきたときは、`ir_log.csv` 以外の名前になっていないか見ます。

??? failure "学習のたびに数字が変わる"
    `keras.utils.set_random_seed(0)` を入れていても、データが少ないと乱数で 1〜2° 変わります。比べるときは、乱数を変えて 3 回ずつやって平均します。

## 発展課題

- 取り直しで分けたテストと、**ランダムに分けた** テストで数字を比べる（2-4）。差が大きいほど、取り直しの条件の違いが大きいということです
- 距離の予測が外れる場面で、ボールの明るさの違い（電池）が効いていないか調べる
