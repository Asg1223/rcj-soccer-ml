// CoreS3：カメラで撮って SD カードに保存する  教材 6-3
//
// 画面をタッチするたびに 1 枚、または「AUTO」にすると 1 秒に 1 枚、/img/ に JPEG で保存する。
// 必要なライブラリ：M5CoreS3、M5Unified（ライブラリマネージャ）。ボード：M5CoreS3
// !!! 撮った画像（特に他チームのロボットが写ったもの）はチーム用リポジトリにだけ置く

#include <M5CoreS3.h>
#include <SD.h>
#include <SPI.h>
#include "esp_camera.h"
#include "img_converters.h"

const int SD_SCK = 36, SD_MISO = 35, SD_MOSI = 37, SD_CS = 4;
const int JPEG_QUALITY = 80;          // 0〜100（大きいほどきれいで、ファイルが大きい）
bool sdOk = false, autoMode = false;
int counter = 0;
uint32_t lastShot = 0;

int nextIndex() {                     // 既にあるファイルの続きの番号から始める
  int i = 0;
  char path[32];
  do {
    snprintf(path, sizeof(path), "/img/img_%05d.jpg", ++i);
  } while (SD.exists(path));
  return i;
}

bool saveFrame(camera_fb_t* fb) {
  uint8_t* jpg = nullptr;
  size_t len = 0;
  if (!frame2jpg(fb, JPEG_QUALITY, &jpg, &len)) return false;
  char path[32];
  snprintf(path, sizeof(path), "/img/img_%05d.jpg", counter);
  File f = SD.open(path, FILE_WRITE);
  bool ok = f && f.write(jpg, len) == len;
  if (f) f.close();
  free(jpg);
  if (ok) counter++;
  return ok;
}

void setup() {
  auto cfg = M5.config();
  CoreS3.begin(cfg);
  if (!CoreS3.Camera.begin()) {
    CoreS3.Display.println("Camera Init Fail");
    while (true) delay(1000);
  }
  CoreS3.Camera.sensor->set_framesize(CoreS3.Camera.sensor, FRAMESIZE_QVGA);   // 320×240
  SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  sdOk = SD.begin(SD_CS, SPI, 25000000);
  if (sdOk) {
    SD.mkdir("/img");
    counter = nextIndex();
  }
}

void loop() {
  CoreS3.update();
  auto t = CoreS3.Touch.getDetail();
  bool shoot = false;
  if (t.wasPressed()) {
    if (t.y > 200 && t.x > 220) autoMode = !autoMode;   // 右下をタッチ：AUTO の切りかえ
    else shoot = true;                                   // それ以外：1 枚撮る
  }
  if (autoMode && millis() - lastShot > 1000) shoot = true;

  if (CoreS3.Camera.get()) {
    camera_fb_t* fb = CoreS3.Camera.fb;
    CoreS3.Display.pushImage(0, 0, CoreS3.Display.width(), CoreS3.Display.height(), (uint16_t*)fb->buf);
    if (shoot && sdOk) {
      bool ok = saveFrame(fb);
      lastShot = millis();
      CoreS3.Display.fillRect(0, 0, 320, 18, ok ? TFT_DARKGREEN : TFT_RED);
    }
    CoreS3.Camera.free();
  }
  CoreS3.Display.setCursor(4, 4);
  CoreS3.Display.setTextColor(TFT_WHITE);
  CoreS3.Display.printf("%s  saved:%d  %s", sdOk ? "SD ok" : "NO SD", counter - 1, autoMode ? "[AUTO]" : "");
  CoreS3.Display.setCursor(240, 220);
  CoreS3.Display.print(autoMode ? "AUTO on " : "AUTO off");
}
