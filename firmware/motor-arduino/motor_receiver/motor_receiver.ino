// モータ側：RCJML_T_MOTOR を受け取って、4 つのモータを回す（Arduino / ESP32）  教材 5-5
//
// モータドライバは「IN1・IN2 の 2 本で向きを決める」型を想定した例。
//   IN1=H, IN2=L：正転　IN1=L, IN2=H：逆転　IN1=L, IN2=L：空転　IN1=H, IN2=H：ブレーキ
// IN1 に PWM、IN2 に普通のデジタル出力を使う（Uno は PWM が出せるピンが 3, 5, 6, 9, 10, 11 しかないため）。
//   正転：IN2=L にして IN1 に PWM（速さ）
//   逆転：IN2=H にして IN1 に「反転した」PWM（IN1 が L の時間が長いほど速い）
// 自分のモータドライバの説明書に合わせて setMotor を書きかえる。
// 使う前に：python -m rcjml.export headers <このフォルダ>  で rcjml_packet.h をコピーする。
// !!! 200 ms 指令が来なければ全部止める（通信が切れたときに暴走しないため）

#include "rcjml_packet.h"

const unsigned long LINK_BAUD = 115200;
// モータごとの {IN1（PWM が出せるピン）, IN2（普通のピン）}
const uint8_t MOTOR_PINS[4][2] = {{3, 4}, {5, 7}, {6, 8}, {9, 12}};
const int SIGN[4] = {1, 1, 1, 1};          // 回る向きが逆のモータは -1 にする

#if defined(ESP32)
  #define LINK Serial2
  const int LINK_RX = 16, LINK_TX = 17;
#else
  #define LINK Serial
#endif

rcjml_parser_t parser;
uint32_t lastCmdMs = 0;

void setMotor(int i, int16_t v) {          // v: -1000〜1000
  v = v * SIGN[i];
  int pwm = map(abs(v), 0, 1000, 0, 255);
  if (v >= 0) { digitalWrite(MOTOR_PINS[i][1], LOW);  analogWrite(MOTOR_PINS[i][0], pwm); }
  else        { digitalWrite(MOTOR_PINS[i][1], HIGH); analogWrite(MOTOR_PINS[i][0], 255 - pwm); }
}

void stopAll() {
  for (int i = 0; i < 4; i++) { digitalWrite(MOTOR_PINS[i][1], LOW); analogWrite(MOTOR_PINS[i][0], 0); }
}

void setup() {
#if defined(ESP32)
  LINK.begin(LINK_BAUD, SERIAL_8N1, LINK_RX, LINK_TX);
#else
  LINK.begin(LINK_BAUD);
#endif
  for (int i = 0; i < 4; i++) { pinMode(MOTOR_PINS[i][0], OUTPUT); pinMode(MOTOR_PINS[i][1], OUTPUT); }
  rcjml_parser_init(&parser);
  stopAll();
}

void loop() {
  while (LINK.available()) {
    if (rcjml_parser_feed(&parser, (uint8_t)LINK.read()) && parser.type == RCJML_T_MOTOR && parser.len == 8) {
      for (int i = 0; i < 4; i++) setMotor(i, rcjml_get_i16(parser.payload + 2 * i));
      lastCmdMs = millis();
    }
  }
  if (millis() - lastCmdMs > 200) stopAll();
}
