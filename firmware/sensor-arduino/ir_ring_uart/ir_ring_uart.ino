// センサ側：IR センサリングを読み、パケットで CoreS3 に送る（Arduino / ESP32）  教材 5-1, 5-2
//
// 送るもの：RCJML_T_IR_RAW（センサの値 uint16 × N）
//   RUN_BALLNET を有効にすると、センサ側で BallNet を動かして RCJML_T_BALL も送る（第4部のモデルを使う）
//
// 使う前に：python -m rcjml.export headers <このフォルダ>  で rcjml_packet.h（と rcjml_mlp.h）をコピーする。
// RUN_BALLNET を使うときは、4-1 の export_to_sketch で ballnet_model.h もこのフォルダに書き出す。
//
// !!! Arduino Uno（5V）の TX を CoreS3（3.3V）の RX に直接つながない。分圧抵抗かレベル変換を入れる（5-1）

// #define RUN_BALLNET

#include "rcjml_packet.h"
#ifdef RUN_BALLNET
#include "ballnet_model.h"
#endif

// ===== 設定（1-2 の ir_ring_reader と同じにする）=====
const uint8_t SENSOR_PINS[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
const unsigned long WINDOW_US = 2500;
const int MAX_READING = 1023;
const bool ACTIVE_LOW = true;
const unsigned long LINK_BAUD = 115200;
#if defined(ESP32)
  #define LINK Serial2                 // ESP32：2 つ目の UART を CoreS3 につなぐ
  const int LINK_RX = 16, LINK_TX = 17;
#else
  #define LINK Serial                  // Uno：0・1 番ピン（USB と共用。書き込むときは配線を外す）
#endif
// ======================================================

const uint8_t N = sizeof(SENSOR_PINS) / sizeof(SENSOR_PINS[0]);
uint16_t values[N];
uint8_t seq = 0;

void readRing() {
  unsigned int low[N];
  for (uint8_t i = 0; i < N; i++) low[i] = 0;
  unsigned int sweeps = 0;
  unsigned long start = micros();
  while (micros() - start < WINDOW_US) {
    for (uint8_t i = 0; i < N; i++) {
      bool on = (digitalRead(SENSOR_PINS[i]) == LOW) == ACTIVE_LOW;
      if (on) low[i]++;
    }
    sweeps++;
  }
  for (uint8_t i = 0; i < N; i++) values[i] = (uint16_t)((unsigned long)low[i] * MAX_READING / sweeps);
}

void sendPacket(uint8_t type, const uint8_t* payload, uint8_t len) {
  uint8_t buf[6 + RCJML_MAX_PAYLOAD];
  uint16_t n = rcjml_pkt_encode(type, seq, payload, len, buf);
  LINK.write(buf, n);
}

void setup() {
#if defined(ESP32)
  Serial.begin(115200);                // USB（デバッグ用）
  LINK.begin(LINK_BAUD, SERIAL_8N1, LINK_RX, LINK_TX);
#else
  LINK.begin(LINK_BAUD);
#endif
  for (uint8_t i = 0; i < N; i++) pinMode(SENSOR_PINS[i], INPUT_PULLUP);
}

void loop() {
  readRing();
  uint8_t payload[2 * N];
  for (uint8_t i = 0; i < N; i++) rcjml_put_u16(payload + 2 * i, values[i]);
  sendPacket(RCJML_T_IR_RAW, payload, 2 * N);

#ifdef RUN_BALLNET
  float x[BALLNET_N_IN], y[BALLNET_N_OUT];
  for (uint8_t i = 0; i < N; i++) x[i] = values[i] * BALLNET_INPUT_SCALE;
  ballnet_forward(x, y);
  rcjml_ball_t b = rcjml_decode_ball(y, BALLNET_DIST_MU, BALLNET_DIST_SIGMA);
  rcjml_ball_msg_t m = {b.angle_deg, b.distance_cm, b.p_hold, b.confidence};
  uint8_t len = rcjml_pack_ball(&m, payload);
  sendPacket(RCJML_T_BALL, payload, len);
#endif
  seq++;
}
