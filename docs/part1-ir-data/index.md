# 第1部 IR データを集める

<span class="ratio">理論 3 : 実装 7</span>

## ねらい

IR センサリングの値を読み、正解（ボールの角度・距離）をつけて PC に保存できるようになります。

## この部の章

| 章 | 内容 |
|---|---|
| [1-1](ball-and-sensor.md) | IR ボールとセンサの仕組み（変調、パルス幅、受光センサの特性） |
| [1-2](read-sensors.md) | センサリングの値を読む：Arduino / ESP32 スケッチ、パルス幅を強さに変換 |
| [1-3](send-to-pc.md) | PC に送る：シリアル通信、Python ロガー、CSV 保存（保存先はチーム用リポジトリ） |
| [1-4](labels.md) | 正解ラベルをつける：角度・距離を印刷したマット、サーボ回転台で自動収集 |
| [1-5](explore.md) | データを眺める：極座標プロット、ヒートマップ、外れ値探し |

## この部で使うシミュレータ

実機がない人は、シミュレータ `rcjml.sim` で同じ流れを体験します。実機で起きる“くせ”を入れてあり、難しさを3段階で選べます。

| 難しさ | どんなデータか | 従来手法の角度の誤差（目安） |
|---|---|---|
| `easy` | くせが少ない。仕組みの理解用 | 約 1° |
| `normal` | 個体差・取り付けのずれ・影・外乱光・反射・遮へいなどが一通り入る | 約 9°（ゴルフボールサイズは約 18°） |
| `hard` | 大会会場を想定。反射・遮へい・ボール以外の赤外線が多く、取り直すたびに条件が大きく変わる | 約 20° |

```python
from rcjml import load_config
from rcjml.sim import make_dataset

cfg = load_config()
df = make_dataset(cfg, n=6000, difficulty="normal", sessions=12)
```

作り済みのデータは `data/ir-synthetic/` にあります（くわしくはそのフォルダの README）。

!!! tip "実機がない人は"
    1-2 と 1-4 の実機の手順は読むだけにして、1-1・1-3・1-5 のシミュレータの手順を進めれば、第2部に進めます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asg1223/rcj-soccer-ml/blob/main/notebooks/01_ir_data.ipynb)
