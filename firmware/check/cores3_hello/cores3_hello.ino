// 0-2 環境チェック用：M5Stack CoreS3
// 画面に文字を出し、画面をタッチした回数を数える。
// 必要なライブラリ：M5Unified（ライブラリマネージャで入れる）

#include <M5Unified.h>

int touches = 0;

void draw() {
  M5.Display.fillScreen(TFT_BLACK);
  M5.Display.setCursor(10, 20);
  M5.Display.setTextColor(TFT_ORANGE);
  M5.Display.setTextSize(3);
  M5.Display.println("RCJ Soccer ML");
  M5.Display.setTextColor(TFT_WHITE);
  M5.Display.setTextSize(2);
  M5.Display.println();
  M5.Display.println(" Touch the screen!");
  M5.Display.printf(" touches = %d\n", touches);
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  Serial.begin(115200);
  draw();
}

void loop() {
  M5.update();
  auto t = M5.Touch.getDetail();
  if (t.wasPressed()) {
    touches++;
    Serial.printf("touch x=%d y=%d\n", t.x, t.y);
    draw();
  }
  delay(10);
}
