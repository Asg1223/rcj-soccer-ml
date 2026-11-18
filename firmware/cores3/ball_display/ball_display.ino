// CoreS3：パケットを受け取り、BallNet で推定して画面に表示する  教材 5-2, 5-3
//
// 受け取る口は 2 つ（どちらからでもよい）:
//   - Port C（UART）：センサ側のマイコン（ir_ring_uart）から
//   - USB：PC から（python -m rcjml.packet send-sim でシミュレータの値を送って試せる）
//
// 使う前に：4-1 の export_to_sketch で、このフォルダに ballnet_model.h・rcjml_mlp.h・rcjml_packet.h を書き出す。

#include <M5Unified.h>
#include "rcjml_packet.h"
#include "ballnet_model.h"

const unsigned long LINK_BAUD = 115200;
const int PORTC_RX = 18, PORTC_TX = 17;   // CoreS3 の Port C。データが来ないときは 2 つを入れかえてみる

rcjml_parser_t usbParser, linkParser;
float values[BALLNET_N_IN];
bool haveValues = false;
rcjml_ball_t ball = {0, 0, 0, 0};
uint32_t frames = 0, lastFpsMs = 0, fps = 0;

void handlePacket(const rcjml_parser_t& p) {
  if (p.type == RCJML_T_IR_RAW && p.len == 2 * BALLNET_N_IN) {
    float x[BALLNET_N_IN], y[BALLNET_N_OUT];
    for (int i = 0; i < BALLNET_N_IN; i++) {
      values[i] = rcjml_get_u16(p.payload + 2 * i);
      x[i] = values[i] * BALLNET_INPUT_SCALE;
    }
    ballnet_forward(x, y);
    ball = rcjml_decode_ball(y, BALLNET_DIST_MU, BALLNET_DIST_SIGMA);
    haveValues = true;
    frames++;
  } else if (p.type == RCJML_T_BALL && p.len == 6) {
    rcjml_ball_msg_t m = rcjml_unpack_ball(p.payload);   // センサ側で推定済み
    ball.angle_deg = m.angle_deg;
    ball.distance_cm = m.distance_cm;
    ball.p_hold = m.p_hold;
    ball.confidence = m.confidence;
    frames++;
  }
}

void pump(Stream& s, rcjml_parser_t& p) {
  while (s.available()) {
    if (rcjml_parser_feed(&p, (uint8_t)s.read())) handlePacket(p);
  }
}

// ロボットを真上から見た図：センサの値の棒と、ボールの方向の矢印
void draw() {
  auto& d = M5.Display;
  const int cx = 110, cy = 120, r0 = 30, r1 = 95;
  d.startWrite();
  d.fillRect(0, 0, 220, 240, TFT_BLACK);
  d.drawCircle(cx, cy, r0, TFT_DARKGREY);
  if (haveValues) {
    for (int i = 0; i < BALLNET_N_IN; i++) {
      float th = (360.0f * i / BALLNET_N_IN) * DEG_TO_RAD;        // 正面（上）が 0°、反時計回り
      float len = r0 + (r1 - r0) * values[i] / 1023.0f;
      int x0 = cx - sinf(th) * r0, y0 = cy - cosf(th) * r0;
      int x1 = cx - sinf(th) * len, y1 = cy - cosf(th) * len;
      d.drawLine(x0, y0, x1, y1, TFT_SKYBLUE);
    }
  }
  float a = ball.angle_deg * DEG_TO_RAD;
  int bx = cx - sinf(a) * (r1 + 10), by = cy - cosf(a) * (r1 + 10);
  d.drawLine(cx, cy, bx, by, TFT_ORANGE);
  d.fillCircle(bx, by, 6, ball.p_hold > 0.5f ? TFT_GREEN : TFT_ORANGE);
  d.endWrite();

  d.fillRect(220, 0, 100, 240, TFT_BLACK);
  d.setCursor(224, 10);
  d.setTextColor(TFT_WHITE);
  d.setTextSize(2);
  d.printf("ANG\n");
  d.setCursor(224, 30);
  d.printf("%5.0f", ball.angle_deg);
  d.setCursor(224, 60);
  d.printf("DIST\n");
  d.setCursor(224, 80);
  d.printf("%5.0f", ball.distance_cm);
  d.setCursor(224, 110);
  d.printf("HOLD\n");
  d.setCursor(224, 130);
  d.printf("%5.2f", ball.p_hold);
  d.setTextSize(1);
  d.setCursor(224, 170);
  d.printf("fps %lu\nok  %lu\ncrc %lu", (unsigned long)fps,
           (unsigned long)(usbParser.ok + linkParser.ok),
           (unsigned long)(usbParser.crc_errors + linkParser.crc_errors));
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  Serial.begin(115200);
  Serial2.begin(LINK_BAUD, SERIAL_8N1, PORTC_RX, PORTC_TX);
  rcjml_parser_init(&usbParser);
  rcjml_parser_init(&linkParser);
  M5.Display.fillScreen(TFT_BLACK);
}

void loop() {
  M5.update();
  pump(Serial, usbParser);
  pump(Serial2, linkParser);
  static uint32_t lastDraw = 0;
  uint32_t now = millis();
  if (now - lastFpsMs >= 1000) {
    fps = frames;
    frames = 0;
    lastFpsMs = now;
  }
  if (now - lastDraw >= 50) {     // 画面の書きかえは 1 秒に 20 回まで（重いので）
    draw();
    lastDraw = now;
  }
}
