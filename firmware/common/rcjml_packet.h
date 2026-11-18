// rcjml_packet.h — マイコン同士の通信のパケット  教材 5-2
//
// 1 つのパケットの形（バイト）:
//   0xA5 0x5A  TYPE  LEN  SEQ  PAYLOAD（LEN バイト）  CRC8
//   ─先頭の印─  種類  長さ 番号  中身                   チェック
//
//   CRC8 は TYPE から PAYLOAD の最後までを、多項式 0x07（CRC-8/SMBUS）で計算したもの。
//   数は全部リトルエンディアン（下位バイトが先）。
//
// Arduino（AVR / ESP32）でも PC（gcc）でも、同じコードで動く。Python 版は rcjml/packet.py。
#pragma once
#include <stdint.h>
#include <string.h>

#define RCJML_SYNC1 0xA5
#define RCJML_SYNC2 0x5A
#define RCJML_MAX_PAYLOAD 64

// パケットの種類
enum {
  RCJML_T_IR_RAW = 0x01,   // センサの値：uint16 × センサ数
  RCJML_T_BALL = 0x02,     // ボールの推定：int16 角度×10, uint16 距離×10, uint8 保持×255, uint8 自信×255
  RCJML_T_MOTOR = 0x10,    // モータの指令：int16 × 4（-1000〜1000）
  RCJML_T_MODE = 0x11,     // 動作モード：uint8
};

static inline uint8_t rcjml_crc8(const uint8_t* data, uint16_t n) {
  uint8_t crc = 0x00;
  for (uint16_t i = 0; i < n; i++) {
    crc ^= data[i];
    for (uint8_t b = 0; b < 8; b++) crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
  }
  return crc;
}

// パケットを作る。out には 6 + len バイト以上の場所が要る。作ったバイト数を返す
static inline uint16_t rcjml_pkt_encode(uint8_t type, uint8_t seq, const uint8_t* payload, uint8_t len,
                                        uint8_t* out) {
  if (len > RCJML_MAX_PAYLOAD) return 0;
  out[0] = RCJML_SYNC1;
  out[1] = RCJML_SYNC2;
  out[2] = type;
  out[3] = len;
  out[4] = seq;
  if (len) memcpy(out + 5, payload, len);
  out[5 + len] = rcjml_crc8(out + 2, (uint16_t)(3 + len));
  return (uint16_t)(6 + len);
}

// 受け取る側：1 バイトずつ渡すと、パケットがそろったときに 1 を返す
typedef struct {
  uint8_t state;
  uint8_t buf[3 + RCJML_MAX_PAYLOAD];   // TYPE, LEN, SEQ, PAYLOAD
  uint8_t pos;
  uint8_t type, len, seq;
  const uint8_t* payload;               // そろったパケットの中身
  uint32_t ok, crc_errors, len_errors;  // 数えておくと通信の調子がわかる
} rcjml_parser_t;

static inline void rcjml_parser_init(rcjml_parser_t* p) {
  memset(p, 0, sizeof(*p));
}

static inline int rcjml_parser_feed(rcjml_parser_t* p, uint8_t c) {
  switch (p->state) {
    case 0:  // 先頭の印 1 つ目を待つ
      if (c == RCJML_SYNC1) p->state = 1;
      return 0;
    case 1:  // 先頭の印 2 つ目
      if (c == RCJML_SYNC2) { p->state = 2; p->pos = 0; }
      else p->state = (c == RCJML_SYNC1) ? 1 : 0;
      return 0;
    case 2:  // TYPE, LEN, SEQ, PAYLOAD を集める
      p->buf[p->pos++] = c;
      if (p->pos == 2 && p->buf[1] > RCJML_MAX_PAYLOAD) {   // 長さがおかしい → 捨てて探し直す
        p->len_errors++;
        p->state = (c == RCJML_SYNC1) ? 1 : 0;
        return 0;
      }
      if (p->pos >= 3 && p->pos == 3 + p->buf[1]) p->state = 3;
      return 0;
    case 3: {  // CRC を確かめる
      p->state = 0;
      if (rcjml_crc8(p->buf, p->pos) != c) { p->crc_errors++; return 0; }
      p->type = p->buf[0];
      p->len = p->buf[1];
      p->seq = p->buf[2];
      p->payload = p->buf + 3;
      p->ok++;
      return 1;
    }
  }
  p->state = 0;
  return 0;
}

// 数の詰め方・取り出し方（リトルエンディアン）
static inline void rcjml_put_u16(uint8_t* b, uint16_t v) { b[0] = (uint8_t)(v & 0xFF); b[1] = (uint8_t)(v >> 8); }
static inline void rcjml_put_i16(uint8_t* b, int16_t v) { rcjml_put_u16(b, (uint16_t)v); }
static inline uint16_t rcjml_get_u16(const uint8_t* b) { return (uint16_t)(b[0] | ((uint16_t)b[1] << 8)); }
static inline int16_t rcjml_get_i16(const uint8_t* b) { return (int16_t)rcjml_get_u16(b); }

// ボールの推定を詰める・取り出す
typedef struct { float angle_deg, distance_cm, p_hold, confidence; } rcjml_ball_msg_t;

static inline uint8_t rcjml_pack_ball(const rcjml_ball_msg_t* m, uint8_t* b) {
  float a = m->angle_deg * 10.0f, d = m->distance_cm * 10.0f;
  float h = m->p_hold * 255.0f, c = m->confidence * 255.0f;
  if (d > 65535.0f) d = 65535.0f;
  if (d < 0.0f) d = 0.0f;
  if (h > 255.0f) h = 255.0f;
  if (h < 0.0f) h = 0.0f;
  if (c > 255.0f) c = 255.0f;
  if (c < 0.0f) c = 0.0f;
  rcjml_put_i16(b, (int16_t)(a + (a >= 0 ? 0.5f : -0.5f)));
  rcjml_put_u16(b + 2, (uint16_t)(d + 0.5f));
  b[4] = (uint8_t)(h + 0.5f);
  b[5] = (uint8_t)(c + 0.5f);
  return 6;
}

static inline rcjml_ball_msg_t rcjml_unpack_ball(const uint8_t* b) {
  rcjml_ball_msg_t m;
  m.angle_deg = rcjml_get_i16(b) / 10.0f;
  m.distance_cm = rcjml_get_u16(b + 2) / 10.0f;
  m.p_hold = b[4] / 255.0f;
  m.confidence = b[5] / 255.0f;
  return m;
}
