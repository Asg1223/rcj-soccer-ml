# 第4部 マイコンに載せる

<span class="ratio">理論 2.5 : 実装 7.5</span>

## ねらい

学習したモデルを Arduino / ESP32 / CoreS3 の上で動かします。

## この部の章

| 章 | 内容 |
|---|---|
| [4-1](export-c.md) | 重みを C ヘッダに書き出し、自作の推論関数で動かす（Arduino Uno でも動く） |
| [4-2](measure.md) | メモリと速度を測る：Uno / ESP32 / CoreS3 の比較 |
| [4-3](quantization.md) | 量子化（float → int8） |
| [4-4](tflite-micro.md) | TensorFlow Lite Micro（LiteRT for Microcontrollers）を ESP32 / CoreS3 で動かす |

## この部のまとめ（BallNet、normal のデータ）

| 動かし方 | 角度 | 大きさ | 動くマイコン |
|---|---|---|---|
| Keras（PC） | 5.77° | — | — |
| 自作の C・float（4-1） | 5.77°（Keras と同じ） | 重み 21,008 バイト | ESP32 / CoreS3（Uno はぎりぎり） |
| 自作の C・重み int8（4-3） | 6.12° | 重み 5,648 バイト | Uno / ESP32 / CoreS3 |
| TFLite Micro・全部 int8（4-4） | 6.20° | 11,432 バイト | ESP32 / CoreS3 |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asg1223/rcj-soccer-ml/blob/main/notebooks/04_on_device.ipynb)
