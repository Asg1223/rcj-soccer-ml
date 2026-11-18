// CoreS3：カメラ画像で物体検出（FOMO 風、TensorFlow Lite Micro）  教材 6-6, 6-7, 6-9
//
// 1. カメラで 320×240 を撮る → 2. 真ん中の 240×240 を 96×96 に縮める → 3. モデルで 12×12 のマスごとに判定
// 4. 見つけた物（ボール・黄ゴール・青ゴール・ロボット）を画面に描き、シリアルに角度を送る
//
// 必要なライブラリ：M5CoreS3、M5Unified、ArduTFLite（Chirale_TensorFlowLite）。ボード：M5CoreS3
// 使う前に：6-6 のノートブックで fomo_tflite.h をこのフォルダに書き出す。
// !!! 自分の画像で学習したモデルは、チーム用リポジトリに置く

#include <M5CoreS3.h>
#include <ArduTFLite.h>
#include "esp_camera.h"
#include "fomo_tflite.h"

const int IN = 96, GRID = 12, N_CLS = 5;          // 出力：マスごとに「なし、ボール、黄、青、ロボット」
const float THRESHOLD = 0.5f;
const float FOV_DEG = 60.0f;                       // カメラの横の画角（自分のカメラで測って直す）
const char* NAMES[N_CLS] = {"none", "ball", "yellow", "blue", "robot"};
const uint16_t COLORS[N_CLS] = {TFT_BLACK, TFT_ORANGE, TFT_YELLOW, TFT_BLUE, TFT_RED};

constexpr int ARENA_SIZE = 200 * 1024;             // CNN は全結合より多くのメモリが要る。足りなければ増やす
uint8_t* arena = nullptr;
float probs[GRID * GRID * N_CLS];

void setup() {
  auto cfg = M5.config();
  CoreS3.begin(cfg);
  Serial.begin(115200);
  if (!CoreS3.Camera.begin()) { CoreS3.Display.println("Camera Init Fail"); while (true) delay(1000); }
  CoreS3.Camera.sensor->set_framesize(CoreS3.Camera.sensor, FRAMESIZE_QVGA);   // 320×240, RGB565
  arena = (uint8_t*)heap_caps_malloc(ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);   // PSRAM に置く
  if (!arena || !modelInit(fomo_tflite, arena, ARENA_SIZE)) {
    CoreS3.Display.println("modelInit failed");
    while (true) delay(1000);
  }
}

// RGB565 の 1 画素を R, G, B（0〜255）にする
static inline void rgb565(uint16_t p, float* r, float* g, float* b) {
  p = (p >> 8) | (p << 8);                          // カメラのバイト順を直す
  *r = ((p >> 11) & 0x1F) * 255.0f / 31.0f;
  *g = ((p >> 5) & 0x3F) * 255.0f / 63.0f;
  *b = (p & 0x1F) * 255.0f / 31.0f;
}

void loop() {
  CoreS3.update();
  if (!CoreS3.Camera.get()) return;
  camera_fb_t* fb = CoreS3.Camera.fb;
  const uint16_t* px = (const uint16_t*)fb->buf;
  const int fw = fb->width, fh = fb->height, side = min(fw, fh), ox = (fw - side) / 2, oy = (fh - side) / 2;

  uint32_t t0 = millis();
  // 2. 縮める（近い画素を 1 つ取る、いちばん簡単な方法）
  int k = 0;
  for (int y = 0; y < IN; y++) {
    int sy = oy + y * side / IN;
    for (int x = 0; x < IN; x++) {
      int sx = ox + x * side / IN;
      float r, g, b;
      rgb565(px[sy * fw + sx], &r, &g, &b);
      modelSetInput(r, k++);                       // モデルの中で 1/255 にする（6-6 の Rescaling）
      modelSetInput(g, k++);
      modelSetInput(b, k++);
    }
  }
  uint32_t t1 = millis();
  modelRunInference();
  uint32_t t2 = millis();
  for (int i = 0; i < GRID * GRID * N_CLS; i++) probs[i] = modelGetOutput(i);

  CoreS3.Display.pushImage(0, 0, fw, fh, (uint16_t*)fb->buf);
  CoreS3.Camera.free();

  // 3〜4. マスごとに、いちばん確率の高い種類を見る
  const float cellPx = (float)side / GRID;
  for (int gy = 0; gy < GRID; gy++) {
    for (int gx = 0; gx < GRID; gx++) {
      const float* p = probs + (gy * GRID + gx) * N_CLS;
      int best = 0;
      for (int c = 1; c < N_CLS; c++) if (p[c] > p[best]) best = c;
      if (best == 0 || p[best] < THRESHOLD) continue;
      int cx = ox + (gx + 0.5f) * cellPx, cy = oy + (gy + 0.5f) * cellPx;
      CoreS3.Display.drawCircle(cx, cy, cellPx / 2, COLORS[best]);
      float x96 = (gx + 0.5f) * IN / GRID;
      float angle = (IN / 2.0f - x96) / (IN / 2.0f) * (FOV_DEG / 2.0f);       // 左が +（6-9）
      Serial.printf("CAM,%lu,%s,%.1f,%.2f\n", millis(), NAMES[best], angle, p[best]);
    }
  }
  CoreS3.Display.setCursor(4, 4);
  CoreS3.Display.setTextColor(TFT_WHITE, TFT_BLACK);
  CoreS3.Display.printf("prep %lu ms  infer %lu ms", t1 - t0, t2 - t1);
}
