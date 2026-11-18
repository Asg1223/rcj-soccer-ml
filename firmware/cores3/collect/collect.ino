// CoreS3：データ収集モード（タッチ画面でラベルを決めて、SD カードに記録）  教材 5-4
//
// センサ側のマイコン（ir_ring_uart）から Port C で RCJML_T_IR_RAW を受け取り、
// 画面で選んだ「角度・距離・保持・取り直し」と一緒に、SD カードの /ir_log.csv に追記する。
// 列は 1-3 のロガーと同じなので、そのまま rcjml.data.load_log で読める。
//
// 使う前に：python -m rcjml.export headers <このフォルダ>  で rcjml_packet.h をコピーする。
// !!! 記録したデータ（SD カード）はチーム用リポジトリにだけ置く

#include <M5Unified.h>
#include <SD.h>
#include <SPI.h>
#include "rcjml_packet.h"

const int N_SENSORS = 12;                 // 自分の機体のセンサの数
const int SAMPLES_PER_SPOT = 20;          // 1 か所で記録する回数
const unsigned long LINK_BAUD = 115200;
const int PORTC_RX = 18, PORTC_TX = 17;
const int SD_SCK = 36, SD_MISO = 35, SD_MOSI = 37, SD_CS = 4;   // CoreS3 の microSD
const char* LOG_PATH = "/ir_log.csv";

rcjml_parser_t parser;
uint16_t latest[N_SENSORS];
uint32_t latestCount = 0;
int angle = 0, distanceCm = 30, session = 1;
bool hold = false;
String status = "ready";

struct Button { int x, y, w, h; const char* label; };
Button buttons[] = {
  {0, 120, 80, 40, "ANG-"}, {80, 120, 80, 40, "ANG+"}, {160, 120, 80, 40, "DST-"}, {240, 120, 80, 40, "DST+"},
  {0, 160, 80, 40, "SES-"}, {80, 160, 80, 40, "SES+"}, {160, 160, 80, 40, "HOLD"}, {240, 160, 80, 40, "REC"},
};
const int N_BUTTONS = sizeof(buttons) / sizeof(buttons[0]);
const int DISTANCES[] = {3, 10, 20, 30, 45, 60, 80, 100, 130, 160};
int distIndex = 3;

void draw() {
  auto& d = M5.Display;
  d.fillScreen(TFT_BLACK);
  d.setTextColor(TFT_WHITE);
  d.setTextSize(2);
  d.setCursor(8, 8);
  if (hold) d.printf("HOLDING the ball\n");
  else d.printf("angle  %+4d deg\n", angle);
  d.setCursor(8, 32);
  d.printf("dist   %4d cm\n", hold ? 3 : distanceCm);
  d.setCursor(8, 56);
  d.printf("session %3d\n", session);
  d.setTextSize(1);
  d.setCursor(8, 84);
  d.printf("packets %lu  crc err %lu\n%s", (unsigned long)latestCount, (unsigned long)parser.crc_errors,
           status.c_str());
  for (int i = 0; i < N_BUTTONS; i++) {
    Button& b = buttons[i];
    uint16_t color = (strcmp(b.label, "REC") == 0) ? TFT_RED : TFT_DARKGREY;
    if (strcmp(b.label, "HOLD") == 0 && hold) color = TFT_GREEN;
    d.fillRoundRect(b.x + 2, b.y + 2, b.w - 4, b.h - 4, 6, color);
    d.setTextSize(2);
    d.setCursor(b.x + 14, b.y + 12);
    d.print(b.label);
  }
}

void pump() {
  while (Serial2.available()) {
    if (rcjml_parser_feed(&parser, (uint8_t)Serial2.read()) && parser.type == RCJML_T_IR_RAW &&
        parser.len == 2 * N_SENSORS) {
      for (int i = 0; i < N_SENSORS; i++) latest[i] = rcjml_get_u16(parser.payload + 2 * i);
      latestCount++;
    }
  }
}

bool record() {
  bool isNew = !SD.exists(LOG_PATH);
  File f = SD.open(LOG_PATH, FILE_APPEND);
  if (!f) return false;
  if (isNew) {
    for (int i = 0; i < N_SENSORS; i++) f.printf("s%d,", i);
    f.println("angle_deg,distance_cm,has_ball,session,time");
  }
  int got = 0;
  uint32_t startCount = latestCount, t0 = millis();
  while (got < SAMPLES_PER_SPOT && millis() - t0 < 5000) {
    pump();
    if (latestCount != startCount) {
      startCount = latestCount;
      for (int i = 0; i < N_SENSORS; i++) f.printf("%u,", latest[i]);
      f.printf("%d,%d,%d,%d,%lu\n", hold ? 0 : angle, hold ? 3 : distanceCm, hold ? 1 : 0, session,
               (unsigned long)millis());
      got++;
    }
  }
  f.close();
  status = String("saved ") + got + " rows";
  return got == SAMPLES_PER_SPOT;
}

void onButton(const char* label) {
  if (!strcmp(label, "ANG-")) angle = (angle <= -165) ? 180 : angle - 15;
  if (!strcmp(label, "ANG+")) angle = (angle >= 180) ? -165 : angle + 15;
  if (!strcmp(label, "DST-") && distIndex > 0) distanceCm = DISTANCES[--distIndex];
  if (!strcmp(label, "DST+") && distIndex < 9) distanceCm = DISTANCES[++distIndex];
  if (!strcmp(label, "SES-") && session > 1) session--;
  if (!strcmp(label, "SES+")) session++;
  if (!strcmp(label, "HOLD")) hold = !hold;
  if (!strcmp(label, "REC")) {
    status = "recording...";
    draw();
    if (!record()) status = "record FAILED (SD / sensor?)";
  }
  draw();
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  Serial2.begin(LINK_BAUD, SERIAL_8N1, PORTC_RX, PORTC_TX);
  rcjml_parser_init(&parser);
  SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  if (!SD.begin(SD_CS, SPI, 25000000)) status = "SD card not found";
  draw();
}

void loop() {
  M5.update();
  pump();
  auto t = M5.Touch.getDetail();
  if (t.wasPressed()) {
    for (int i = 0; i < N_BUTTONS; i++) {
      Button& b = buttons[i];
      if (t.x >= b.x && t.x < b.x + b.w && t.y >= b.y && t.y < b.y + b.h) onButton(b.label);
    }
  }
  static uint32_t last = 0;
  if (millis() - last > 500) {      // 受信の数だけ、ときどき書きかえる
    last = millis();
    M5.Display.fillRect(0, 84, 320, 30, TFT_BLACK);
    M5.Display.setTextSize(1);
    M5.Display.setCursor(8, 84);
    M5.Display.printf("packets %lu  crc err %lu\n%s", (unsigned long)latestCount,
                      (unsigned long)parser.crc_errors, status.c_str());
  }
}
