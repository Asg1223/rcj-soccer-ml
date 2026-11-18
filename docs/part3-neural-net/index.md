# 第3部 ニューラルネット：距離と保持

<span class="ratio">理論 3.5 : 実装 6.5</span>

## ねらい

ニューラルネットで、角度に加えて距離とボール保持を推定します。

## この部の章

| 章 | 内容 |
|---|---|
| [3-1](neurons.md) | ニューロンと層（ブラウザデモで重みを動かす） |
| [3-2](keras-mlp.md) | Keras で MLP：角度と距離を同時に推定 |
| [3-3](overfitting.md) | 過学習とその対策 |
| [3-4](classification.md) | 分類：ボール保持判定、混同行列、しきい値 |
| [3-5](augmentation.md) | データ拡張：センサリングの回転対称性を使う |

## この部のまとめ（normal のデータ、テスト用は取り直しごとに分けたもの）

| 方法 | 角度 | 距離（割合の真ん中） | 保持の正解率 | 大きさ |
|---|---|---|---|---|
| 重み付きベクトル和（2-1） | 9.4° | — | — | 重み 24 個 |
| ランダムフォレスト（2-5） | 4.3° | — | — | 約 90 MB |
| BallNet（3-2） | 5.8° | 15% | 99.5% | 約 21 KB |
| BallNet ＋ シミュレータで増やしたデータ（3-3） | 5.2° | — | — | 約 21 KB |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asg1223/rcj-soccer-ml/blob/main/notebooks/03_neural_net.ipynb)

ローカルで動かすときは、先に `pip install tensorflow` を実行します（Python 3.12）。
