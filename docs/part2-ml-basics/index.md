# 第2部 機械学習の基本：角度推定

<span class="ratio">理論 4 : 実装 6</span>

## ねらい

従来の計算方法と機械学習を比べながら、モデルの作り方・評価のしかたを身につけます。

## この部の章

| 章 | 内容 |
|---|---|
| [2-1](baseline.md) | 従来手法（重み付きベクトル和）をまず実装し、ベースラインにする |
| [2-2](linear-regression.md) | 線形回帰と勾配降下（微分は図で説明）を NumPy で自作 |
| [2-3](angle-ring.md) | 角度は輪っか：359° と 1° の問題、sin / cos で出力する理由 |
| [2-4](evaluation.md) | 評価：学習用とテスト用の分け方、MAE、撮影セッションごとに分ける |
| [2-5](scikit-learn.md) | 同じことを scikit-learn で数行に |

## この部のまとめ（normal のデータ、角度の平均誤差）

| 方法 | 誤差 | 章 |
|---|---|---|
| 重み付きベクトル和（従来の方法） | 9.4° | 2-1 |
| 線形回帰（自作・scikit-learn） | 8.9° | 2-2, 2-5 |
| k 近傍法 | 5.0° | 2-5 |
| ランダムフォレスト | 4.3° | 2-5 |

成績はすべて、取り直しごとに分けたテスト用データ（2-4）で測っています。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asg1223/rcj-soccer-ml/blob/main/notebooks/02_angle_estimation.ipynb)
