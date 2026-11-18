// BallNet をマイコンで動かす（Arduino / ESP32 共通）  教材 4-1
//
// IR センサリングを読み（1-2 と同じ方法）、BallNet で角度・距離・保持を推定して送る。
//   BALL,<時刻ms>,<角度°>,<距離cm>,<保持の確率>,<自信>,<推論にかかった µs>
//
// 使う前に：このフォルダごとチーム用リポジトリにコピーし、ノートブック（4-1）の
// export_to_sketch で ballnet_model.h と rcjml_mlp.h をこのフォルダに書き出す。

#include "ballnet_model.h"

// ===== 設定（1-2 の ir_ring_reader と同じにする）=====
const uint8_t SENSOR_PINS[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
const unsigned long WINDOW_US = 2500;
const int MAX_READING = 1023;
const bool ACTIVE_LOW = true;
const unsigned long BAUD = 115200;
// ======================================================

const uint8_t N = sizeof(SENSOR_PINS) / sizeof(SENSOR_PINS[0]);
#if defined(__cplusplus)
static_assert(sizeof(SENSOR_PINS) / sizeof(SENSOR_PINS[0]) == BALLNET_N_IN,
              "センサの数とモデルの入力の数が違います");
#endif

void readRing(float* x) {
  unsigned int low[BALLNET_N_IN] = {0};
  unsigned int sweeps = 0;
  unsigned long start = micros();
  while (micros() - start < WINDOW_US) {
    for (uint8_t i = 0; i < N; i++) {
      bool on = (digitalRead(SENSOR_PINS[i]) == LOW) == ACTIVE_LOW;
      if (on) low[i]++;
    }
    sweeps++;
  }
  for (uint8_t i = 0; i < N; i++) {
    float v = (float)low[i] * MAX_READING / sweeps;   // 0〜1023（1-2 と同じ値）
    x[i] = v * BALLNET_INPUT_SCALE;                    // 0〜1 にそろえる（学習のときと同じ）
  }
}

void setup() {
  Serial.begin(BAUD);
  for (uint8_t i = 0; i < N; i++) pinMode(SENSOR_PINS[i], INPUT_PULLUP);
}

void loop() {
  float x[BALLNET_N_IN], y[BALLNET_N_OUT];
  readRing(x);
  unsigned long t0 = micros();
  ballnet_forward(x, y);
  unsigned long dt = micros() - t0;
  rcjml_ball_t b = rcjml_decode_ball(y, BALLNET_DIST_MU, BALLNET_DIST_SIGMA);

  Serial.print("BALL,");
  Serial.print(millis());
  Serial.print(',');
  Serial.print(b.angle_deg, 1);
  Serial.print(',');
  Serial.print(b.distance_cm, 1);
  Serial.print(',');
  Serial.print(b.p_hold, 3);
  Serial.print(',');
  Serial.print(b.confidence, 3);
  Serial.print(',');
  Serial.println(dt);
}
