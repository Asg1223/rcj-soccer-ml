# 第6部 カメラ認識：ボール・ゴール・相手ロボット

<span class="ratio">理論 3.5 : 実装 6.5</span>

## ねらい

CoreS3 のカメラで、ボール・黄ゴール・青ゴール・相手ロボットを見つけます。

## この部の章

| 章 | 内容 |
|---|---|
| [6-1](pixels.md) | 画像はただの数字の表：画素、RGB と HSV、解像度とフレームレート |
| [6-2](hsv-baseline.md) | 従来手法（HSV しきい値）でボールとゴールを検出。相手ロボットは色では見つけられないことを確かめる |
| [6-3](collect-images.md) | 画像を集める：CoreS3 で撮影し、SD で PC へ |
| [6-4](labeling.md) | ラベル付け：4 種類（ボール／黄ゴール／青ゴール／ロボット）に枠をつける |
| [6-5](cnn.md) | CNN の仕組みと、小さな CNN での分類 |
| [6-6](detection.md) | 物体検出：FOMO 風の検出モデルを作って CoreS3 用に書き出す |
| [6-7](speed.md) | CoreS3 で速く動かす：解像度、切り取り、FPS と遅延 |
| [6-8](robots.md) | 相手ロボットを見つける：知らないチームで評価する、集め方、誤検出の減らし方 |
| [6-9](fusion.md) | IR とカメラの融合と使い分け |

## この部の結果のまとめ

| 章 | やったこと | 結果（シミュレータの画像） |
|---|---|---|
| 6-2 | 色のしきい値 | ボール 77% 見つけた・ロボット 0% |
| 6-5 | 小さな CNN で左・真ん中・右・なしの分類 | 正解率 94%（色のしきい値は 69%） |
| 6-6 | FOMO 風の検出 | ボール 98%・ロボット 91% 見つけた |
| 6-8 | 知らない 10 チームのロボット | 1 チームで学習 65% → 8 チームで 80% |
| 6-9 | IR とカメラを混ぜる | 前のボールの角度のずれ 5.3° → 1.8° |

## この部で使うスケッチ

| スケッチ | 章 |
|---|---|
| `firmware/cores3/camera_capture` | 6-3 |
| `firmware/cores3/camera_detect` | 6-6, 6-7 |

!!! private "画像とモデルは非公開"
    自分で撮った画像、知り合いのチームのロボットの画像、それで学習したモデル（`fomo_tflite.h` など）は、非公開リポジトリだけに置きます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asg1223/rcj-soccer-ml/blob/main/notebooks/06_camera.ipynb)
