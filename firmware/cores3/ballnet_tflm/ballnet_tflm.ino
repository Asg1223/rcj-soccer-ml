// BallNet を TensorFlow Lite Micro で動かす（M5Stack CoreS3 / ESP32）  教材 4-4
//
// 必要なライブラリ（Arduino IDE のライブラリマネージャで入れる）:
//   ArduTFLite（Chirale_TensorFlowLite も一緒に入る）、M5Unified
// 使う前に：ノートブック（4-4）で ballnet_tflite.h をこのフォルダに書き出す。
//
// ここではセンサの代わりに、シリアルから「IR,<ms>,s0,...,s11」の行（1-2 と同じ形）を受け取って推論する。
// 第5部で、センサ側のマイコンからの UART に置きかえる。

#include <M5Unified.h>
#include <ArduTFLite.h>
#include "ballnet_tflite.h"

const int N_IN = 12;
const float INPUT_SCALE = 1.0f / 1023.0f;
const float DIST_MU = 3.4f, DIST_SIGMA = 1.0f;   // rcjml.nn と同じ値

constexpr int ARENA_SIZE = 16 * 1024;            // 足りないと modelInit が失敗する。そのときは増やす
alignas(16) byte arena[ARENA_SIZE];

bool parseLine(const String& line, float* x) {
  if (!line.startsWith("IR,")) return false;
  int pos = line.indexOf(',', 3);                // 時刻の後ろ
  for (int i = 0; i < N_IN; i++) {
    if (pos < 0) return false;
    int next = line.indexOf(',', pos + 1);
    String v = (next < 0) ? line.substring(pos + 1) : line.substring(pos + 1, next);
    x[i] = v.toFloat() * INPUT_SCALE;
    pos = next;
  }
  return true;
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  Serial.begin(115200);
  M5.Display.setTextSize(2);
  if (!modelInit(ballnet_tflite, arena, ARENA_SIZE)) {
    M5.Display.println("modelInit failed");
    Serial.println("# modelInit failed (ARENA_SIZE を増やす)");
    while (true) delay(1000);
  }
  M5.Display.println("BallNet (TFLM) ready");
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  float x[N_IN];
  if (!parseLine(line, x)) return;

  unsigned long t0 = micros();
  for (int i = 0; i < N_IN; i++) modelSetInput(x[i], i);
  modelRunInference();
  float y[4];
  for (int i = 0; i < 4; i++) y[i] = modelGetOutput(i);
  unsigned long dt = micros() - t0;

  float angle = atan2f(y[1], y[0]) * 57.29578f;
  float dist = expf(y[2] * DIST_SIGMA + DIST_MU);
  float p_hold = 1.0f / (1.0f + expf(-y[3]));
  Serial.printf("BALL,%lu,%.1f,%.1f,%.3f,%lu\n", millis(), angle, dist, p_hold, dt);

  M5.Display.fillRect(0, 40, 320, 80, TFT_BLACK);
  M5.Display.setCursor(0, 40);
  M5.Display.printf("angle %6.1f deg\ndist  %6.1f cm\nhold  %5.2f\n%lu us", angle, dist, p_hold, dt);
}
