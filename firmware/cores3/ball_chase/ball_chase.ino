// CoreS3：ボールを追いかける（回りこみ）と、モータ側のマイコンへの指令  教材 5-5
//
// 受け取る：Port C から RCJML_T_IR_RAW（センサ側）→ BallNet で推定（5-3 と同じ）
// 送る    ：Port B から RCJML_T_MOTOR（4 輪の速さ -1000〜1000）→ motor_receiver
//
// 使う前に：4-1 の export_to_sketch で、このフォルダに ballnet_model.h・rcjml_mlp.h・rcjml_packet.h を書き出す。
// !!! 最初はタイヤを床から浮かせて、向きと速さが正しいかを確かめる（5-5 の「安全に試す」）

#include <M5Unified.h>
#include "rcjml_packet.h"
#include "ballnet_model.h"

const unsigned long LINK_BAUD = 115200;
const int PORTC_RX = 18, PORTC_TX = 17;   // センサ側から
const int PORTB_RX = 9, PORTB_TX = 8;     // モータ側へ（つなぐ前に CoreS3 の説明書でピンを確かめる）
const float MAX_SPEED = 600.0f;           // 最初は小さめに（-1000〜1000 のうち）
// 4 つのタイヤの位置（正面から反時計回りの角度）。自分の機体に合わせる
const float WHEEL_DEG[4] = {45.0f, 135.0f, 225.0f, 315.0f};

rcjml_parser_t parser;
rcjml_ball_t ball = {0, 100, 0, 0};
uint32_t lastBallMs = 0;
uint8_t seq = 0;
bool running = false;

// 5-5 の chase_rule と同じ考え方：正面なら直進、それ以外は外側にずらして回りこむ
float chaseDirection(float a, float dist, bool holding, float* speedScale) {
  *speedScale = 1.0f;
  if (holding) return 0.0f;                 // 保持したら正面（相手ゴールの方）へ。第6部でゴールの方向に直す
  if (fabsf(a) < 15.0f) return a;
  float nearGain = constrain(40.0f / fmaxf(dist, 1.0f), 0.3f, 1.0f);
  float offset = copysignf(fminf(fabsf(a), 90.0f), a) * nearGain;
  if (dist < 25.0f) *speedScale = 0.7f;
  float d = a + offset;
  while (d > 180.0f) d -= 360.0f;
  while (d < -180.0f) d += 360.0f;
  return d;
}

// 進む向き（正面 0°、左 +）と速さから、4 つのタイヤの速さを決める（オムニホイール）
void omni4(float directionDeg, float speed, float rotation, int16_t* out) {
  for (int i = 0; i < 4; i++) {
    float v = speed * sinf((directionDeg - WHEEL_DEG[i]) * DEG_TO_RAD) + rotation;
    out[i] = (int16_t)constrain(v, -1000.0f, 1000.0f);
  }
}

void sendMotor(const int16_t* m) {
  uint8_t payload[8], buf[6 + 8];
  for (int i = 0; i < 4; i++) rcjml_put_i16(payload + 2 * i, m[i]);
  uint16_t n = rcjml_pkt_encode(RCJML_T_MOTOR, seq++, payload, 8, buf);
  Serial1.write(buf, n);
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  Serial2.begin(LINK_BAUD, SERIAL_8N1, PORTC_RX, PORTC_TX);
  Serial1.begin(LINK_BAUD, SERIAL_8N1, PORTB_RX, PORTB_TX);
  rcjml_parser_init(&parser);
  M5.Display.setTextSize(2);
  M5.Display.println("touch to START / STOP");
}

void loop() {
  M5.update();
  while (Serial2.available()) {
    if (rcjml_parser_feed(&parser, (uint8_t)Serial2.read()) && parser.type == RCJML_T_IR_RAW &&
        parser.len == 2 * BALLNET_N_IN) {
      float x[BALLNET_N_IN], y[BALLNET_N_OUT];
      for (int i = 0; i < BALLNET_N_IN; i++) x[i] = rcjml_get_u16(parser.payload + 2 * i) * BALLNET_INPUT_SCALE;
      ballnet_forward(x, y);
      ball = rcjml_decode_ball(y, BALLNET_DIST_MU, BALLNET_DIST_SIGMA);
      lastBallMs = millis();
    }
  }
  if (M5.Touch.getDetail().wasPressed()) running = !running;

  int16_t motors[4] = {0, 0, 0, 0};
  bool fresh = millis() - lastBallMs < 100;     // 100 ms 以上データが来なければ止まる（安全のため）
  if (running && fresh && ball.confidence > 0.3f) {
    float scale;
    float dir = chaseDirection(ball.angle_deg, ball.distance_cm, ball.p_hold > 0.9f, &scale);
    omni4(dir, MAX_SPEED * scale, 0.0f, motors);
  }
  sendMotor(motors);

  static uint32_t last = 0;
  if (millis() - last > 100) {
    last = millis();
    M5.Display.fillRect(0, 30, 320, 120, TFT_BLACK);
    M5.Display.setCursor(0, 30);
    M5.Display.printf("%s\nball %5.0f deg %4.0f cm\nmotor %d %d %d %d", running ? "RUNNING" : "STOPPED",
                      ball.angle_deg, ball.distance_cm, motors[0], motors[1], motors[2], motors[3]);
  }
  delay(5);
}
