# H8 CoreS3 で物体を見つける

| 時間の目安 | 本文 | 使う機材 |
|---|---|---|
| 90〜120 分 | [6-5](../part6-camera/cnn.md)、[6-6](../part6-camera/detection.md)、[6-7](../part6-camera/speed.md)、[6-8](../part6-camera/robots.md) | PC（できれば Colab の GPU）、CoreS3（ロボットに付けた状態） |

## ゴール

- H7 の画像で、FOMO 風の検出モデルを学習する
- **別の撮影フォルダ** で、4 種類それぞれの「見つけた割合」と「当たった割合」を測る
- CoreS3 で動かし、1 秒に何枚処理できるかを測る

---

## ステップ

### 1. 学習して評価する（40 分）

チーム用リポジトリの `notebooks/` に新しいノートブックを作ります。CPU だと 10 分以上かかることがあります。

```python
import numpy as np
import keras
from rcjml import load_config
from rcjml.vision import (load_yolo_folder, build_fomo, fomo_loss, grid_targets,
                          fomo_detections, evaluate, augment_images)

cfg = load_config()
team = cfg.source.parent.parent
X, labels, sessions = load_yolo_folder(team / "data" / "camera",
                                       class_names=["ball", "yellow_goal", "blue_goal", "robot"])
sessions = np.array(sessions)
names = sorted(set(sessions.tolist()))
test_mask = sessions == names[-1]                 # 最後に撮ったフォルダをテストに
print("フォルダ:", names, " 学習", (~test_mask).sum(), "枚 / テスト", test_mask.sum(), "枚")

X_tr = X[~test_mask]
L_tr = [l for l, t in zip(labels, test_mask) if not t]
X_te = X[test_mask]
L_te = [l for l, t in zip(labels, test_mask) if t]
X_aug = np.concatenate([X_tr, augment_images(X_tr, np.random.default_rng(0))])   # 明るさ・色・ノイズ
L_aug = L_tr + L_tr

keras.utils.set_random_seed(0)
fomo = build_fomo()
fomo.compile(optimizer=keras.optimizers.Adam(2e-3), loss=fomo_loss())
fomo.fit(X_aug, grid_targets(L_aug), validation_split=0.15, epochs=40, batch_size=32, verbose=0,
         callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])

P = fomo.predict(X_te, verbose=0)
r = evaluate([fomo_detections(p, 0.5) for p in P], L_te)
for name, v in r.items():
    print(f"{name:12s} 見つけた {v['recall']:.0%}  当たった {v['precision']:.0%}  （正解 {v['n_true']} 個）")
```

- [ ] 学習とテストに **別の撮影フォルダ** が入っていることを確かめた
- [ ] 4 種類の「見つけた」「当たった」を記録シートに書いた

### 2. 見のがしを見る（20 分）

6-6 の「手を動かす 3」と同じように、テストの画像に正解の枠と見つけた点を重ねて表示します。

- [ ] 見のがした画像を 10 枚見て、多い理由を記録シートに書いた（遠い？ 暗い？ 隠れている？ 知らないロボット？）
- [ ] その見え方の画像を H7 に戻って撮り足す計画を立てた

### 3. マイコン用に書き出す（15 分）

- [ ] `firmware/cores3/camera_detect` を、チーム用リポジトリの `firmware/` にコピーした

```python
import tensorflow as tf
from rcjml.export import tflite_to_c

def representative_images():
    for i in range(0, min(len(X_tr), 400), 4):
        yield [X_tr[i:i + 1].astype(np.float32)]

conv = tf.lite.TFLiteConverter.from_keras_model(fomo)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representative_images
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
fomo_int8 = conv.convert()
print("int8 のモデル:", len(fomo_int8), "バイト")

sketch = team / "firmware" / "camera_detect"        # camera_detect をコピーしたフォルダ
print(tflite_to_c(fomo_int8, sketch / "fomo_tflite.h", name="fomo_tflite"))
```

- [ ] チーム用リポジトリの `firmware/camera_detect/` に `fomo_tflite.h` ができた

### 4. CoreS3 で動かす（25 分）

- [ ] 4-4 の ArduTFLite と M5CoreS3 のライブラリを入れて、`camera_detect.ino` を書き込んだ
- [ ] 画面に映像が出て、ボールの位置に丸が描かれた
- [ ] シリアルに `CAM,時間(ms),種類,角度,確率` が出た
- [ ] ボールを左に動かすと、角度が **＋** に増えた（左が ＋）
- [ ] 1 秒に何枚処理できるか（`CAM` の行の時間の間隔から）を記録した

---

## 合格条件

- [ ] 別の撮影フォルダのテストで、ボールを **80% 以上** 見つけられる
- [ ] 黄ゴール・青ゴールを、それぞれ 80% 以上見つけられる
- [ ] ロボットの「見つけた」「当たった」を測り、知っているロボットと知らないロボットで違うかを確かめた（6-8）
- [ ] CoreS3 で **1 秒に 5 枚以上** 処理できる（足りないときは 6-7 の方法で速くする）
- [ ] `fomo_tflite.h` はチーム用リポジトリにだけある

## 記録シート

| 種類 | 見つけた | 当たった | 正解の数 | 見のがしの多い理由 |
|---|---|---|---|---|
| ball | | | | |
| yellow_goal | | | | |
| blue_goal | | | | |
| robot | | | | |

| 項目 | 記録 |
|---|---|
| 学習に使ったフォルダ / テストのフォルダ | |
| int8 のモデルの大きさ（バイト） | |
| CoreS3 の 1 枚の処理時間（ms）・1 秒の枚数 | |
| 撮り足す見え方 | |

---

## 困ったとき

??? failure "PC では見つかるのに、CoreS3 ではほとんど見つからない"
    CoreS3 での画像の切り取り方・縮め方・色の並び（RGB565 → RGB）が、学習のときと違うかもしれません。CoreS3 で撮った画像（H7 の `camera_capture`）で学習しているか、`camera_detect` の中央切り取りの設定が学習の画像と同じかを確かめます（6-7）。

??? failure "ボールは見つかるが、ロボットがほとんど見つからない"
    ロボットの画像が少ないか、種類が少ないです。チームの数を増やすのがいちばん効きます（6-8）。

??? failure "1 秒に 5 枚に届かない"
    カメラの解像度を下げる、モデルの幅（`build_fomo(width=8)`）を小さくする、画面の描画を間引く、を試します（6-7）。

## 発展課題

- Edge Impulse の FOMO（6-6 のルート A、**プロジェクトは必ず Private**）でも学習し、成績と速さを比べる
- しきい値（`fomo_detections(p, 0.5)` の 0.5）を変えて、「見つけた」と「当たった」の関係を表にする
