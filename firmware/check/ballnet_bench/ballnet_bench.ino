// BallNet の速さとメモリを測る  教材 4-2
// センサはつながなくてよい。決まった入力で 1,000 回推論して、1 回あたりの時間を出す。
// 使う前に：export_to_sketch で ballnet_model.h と rcjml_mlp.h をこのフォルダに書き出す。

#include "ballnet_model.h"

#if defined(ESP32)
size_t freeMemory() { return ESP.getFreeHeap(); }
#elif defined(__AVR__)
extern char *__brkval;
extern char __heap_start;
size_t freeMemory() {
  char top;
  return __brkval ? &top - __brkval : &top - &__heap_start;
}
#else
size_t freeMemory() { return 0; }
#endif

void setup() {
  Serial.begin(115200);
  delay(1500);
  float x[BALLNET_N_IN], y[BALLNET_N_OUT];
  for (int i = 0; i < BALLNET_N_IN; i++) x[i] = 0.1f * (i % 7);

  const int REPEAT = 1000;
  unsigned long t0 = micros();
  for (int k = 0; k < REPEAT; k++) {
    x[k % BALLNET_N_IN] += 1e-6f;       // 毎回少し変えて、計算が省略されないようにする
    ballnet_forward(x, y);
  }
  unsigned long dt = micros() - t0;

  Serial.println("# BallNet bench");
  Serial.print("layers      : ");
  Serial.println(BALLNET_N_LAYERS);
  Serial.print("us / infer  : ");
  Serial.println((float)dt / REPEAT, 1);
  Serial.print("free memory : ");
  Serial.println((unsigned long)freeMemory());
  Serial.print("output[0]   : ");
  Serial.println(y[0], 6);
}

void loop() {}
