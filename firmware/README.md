# firmware

| フォルダ | 中身 | 使う章 |
|---|---|---|
| `check/serial_hello` | Arduino / ESP32 の動作確認 | 0-2 |
| `check/cores3_hello` | M5Stack CoreS3 の動作確認 | 0-2 |
| `sensor-arduino/ir_ring_reader` | IR センサリングの読み取り（Arduino 版） | 1-2 |
| `sensor-esp32/ir_ring_reader` | IR センサリングの読み取り（ESP32 版） | 1-2 |
| `cores3` | カメラ・推論・表示・通信（CoreS3） | 第4〜6部 |

センサの個数やピン番号は、スケッチの中の設定部分で変えます。自分の機体の値をこのリポジトリにコミットしないでください（チーム用リポジトリに置きます）。
