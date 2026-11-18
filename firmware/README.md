# firmware

| フォルダ | 中身 | 使う章 |
|---|---|---|
| `check/serial_hello` | Arduino / ESP32 の動作確認 | 0-2 |
| `check/cores3_hello` | M5Stack CoreS3 の動作確認 | 0-2 |
| `sensor-arduino/ir_ring_reader` | IR センサリングの読み取り（Arduino 版） | 1-2 |
| `sensor-esp32/ir_ring_reader` | IR センサリングの読み取り（ESP32 版） | 1-2 |
| `common` | 共通のヘッダ（`rcjml_mlp.h`：推論、`rcjml_packet.h`：通信）。使うスケッチのフォルダにコピーする | 第4〜5部 |
| `sensor-arduino/ballnet_infer` | センサ側で BallNet を動かす | 4-1 |
| `check/ballnet_bench` | 推論の速さとメモリを測る | 4-2 |
| `cores3/ballnet_tflm` | CoreS3 で TensorFlow Lite Micro | 4-4 |
| `sensor-arduino/ir_ring_uart` | センサの値をパケットで CoreS3 へ送る | 5-1, 5-2 |
| `cores3/ball_display` | パケットを受け取り、推定して表示 | 5-3 |
| `cores3/collect` | タッチ画面でラベルを決め、SD に記録 | 5-4 |
| `cores3/ball_chase` | 回りこみの動きを決め、モータ側へ指令 | 5-5 |
| `motor-arduino/motor_receiver` | モータ側：指令を受けてモータを回す | 5-5 |
| `cores3/camera_capture` | CoreS3 のカメラで撮って SD に保存 | 6-3 |
| `cores3/camera_detect` | カメラ画像で物体検出（TFLM） | 6-6, 6-7 |

Arduino IDE は、スケッチのフォルダの外にあるヘッダを読めません。`common` のヘッダは、`python -m rcjml.export headers <スケッチのフォルダ>` でコピーしてから使います（コピーしたものは `.gitignore` で除外されています）。

センサの個数やピン番号は、スケッチの中の設定部分で変えます。自分の機体の値をこのリポジトリにコミットしないでください（チーム用リポジトリに置きます）。
