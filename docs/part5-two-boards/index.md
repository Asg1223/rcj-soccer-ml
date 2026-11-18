# 第5部 2台をつなぐ：Arduino / ESP32 ↔ CoreS3

<span class="ratio">理論 1.5 : 実装 8.5</span>

## ねらい

IR センサ側のマイコンと CoreS3 をつなぎ、推論結果でロボットを動かします。

## この部の章

| 章 | 内容 |
|---|---|
| [5-1](design.md) | 役割分担の設計 |
| [5-2](packet.md) | UART 通信とパケット設計（ヘッダ、チェックサム、取りこぼし対策） |
| [5-3](display.md) | CoreS3 の画面にボールの方向と距離を表示 |
| [5-4](collect.md) | データ収集モード：CoreS3 のタッチ画面でラベル付け、SD に保存 |
| [5-5](control.md) | 制御ループへの組み込み：推論結果でボールを追いかける |

## この部で使うスケッチ

| スケッチ | マイコン | 章 |
|---|---|---|
| `firmware/sensor-arduino/ir_ring_uart` | センサ側（Arduino / ESP32） | 5-1, 5-2 |
| `firmware/cores3/ball_display` | CoreS3 | 5-3 |
| `firmware/cores3/collect` | CoreS3 | 5-4 |
| `firmware/cores3/ball_chase` | CoreS3 | 5-5 |
| `firmware/motor-arduino/motor_receiver` | モータ側（Arduino / ESP32） | 5-5 |

共通のヘッダ（`rcjml_packet.h` など）は、`python -m rcjml.export headers <スケッチのフォルダ>` でコピーしてから使います。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asg1223/rcj-soccer-ml/blob/main/notebooks/05_two_boards.ipynb)
