// rcjml_mlp.h — 全結合ニューラルネット（MLP）の推論  教材 4-1, 4-3
//
// Arduino（AVR / ESP32）でも PC（gcc）でも、同じコードで動く。
// 重みは rcjml.export で書き出したヘッダ（例：ballnet_model.h）に入っている。
//
// 使い方:
//   #include "ballnet_model.h"     // 書き出した重み（この中で rcjml_mlp.h も読み込まれる）
//   float in[BALLNET_N_IN], out[BALLNET_N_OUT];
//   ballnet_forward(in, out);      // センサの値（0〜1 にそろえたもの）→ 出力
//
#pragma once
#include <math.h>
#include <stdint.h>

#if defined(__AVR__)
  #include <avr/pgmspace.h>
  #define RCJML_PROGMEM PROGMEM
  static inline float rcjml_read_f(const float* p) { return pgm_read_float(p); }
  static inline int8_t rcjml_read_q(const int8_t* p) { return (int8_t)pgm_read_byte(p); }
#else
  #define RCJML_PROGMEM
  static inline float rcjml_read_f(const float* p) { return *p; }
  static inline int8_t rcjml_read_q(const int8_t* p) { return *p; }
#endif

enum { RCJML_LINEAR = 0, RCJML_RELU = 1 };

// 1 層分の情報。w は「出力 × 入力」の順に並んでいる（w[o * n_in + i]）
typedef struct {
  uint16_t n_in, n_out;
  const float* w;      // float の重み（量子化しないとき）
  const int8_t* wq;    // int8 の重み（量子化したとき。使わなければ 0）
  float w_scale;       // int8 の重み 1 あたりの大きさ
  const float* b;      // バイアス
  uint8_t act;         // RCJML_LINEAR か RCJML_RELU
} rcjml_layer_t;

// 1 層分の計算：out = act(W x + b)
static inline void rcjml_dense(const rcjml_layer_t* L, const float* x, float* out) {
  for (uint16_t o = 0; o < L->n_out; o++) {
    float acc = 0.0f;
    if (L->wq) {
      const int8_t* row = L->wq + (uint32_t)o * L->n_in;
      for (uint16_t i = 0; i < L->n_in; i++) acc += (float)rcjml_read_q(row + i) * x[i];
      acc *= L->w_scale;
    } else {
      const float* row = L->w + (uint32_t)o * L->n_in;
      for (uint16_t i = 0; i < L->n_in; i++) acc += rcjml_read_f(row + i) * x[i];
    }
    acc += rcjml_read_f(L->b + o);
    out[o] = (L->act == RCJML_RELU && acc < 0.0f) ? 0.0f : acc;
  }
}

// 全部の層を順に計算する。buf_a, buf_b はいちばん大きい層の幅以上の作業用配列
static inline void rcjml_forward(const rcjml_layer_t* layers, uint8_t n_layers,
                                 const float* in, float* out, float* buf_a, float* buf_b) {
  const float* x = in;
  for (uint8_t k = 0; k < n_layers; k++) {
    float* y = (k == n_layers - 1) ? out : ((k % 2 == 0) ? buf_a : buf_b);
    rcjml_dense(&layers[k], x, y);
    x = y;
  }
}

// ボールの推定結果（BallNet の出力を、角度・距離・保持の確率に直したもの）
typedef struct {
  float angle_deg;    // -180〜180（正面 0°、左が +）
  float distance_cm;
  float p_hold;       // ボールを保持している確率 0〜1
  float confidence;   // cos・sin の長さ（小さいほど自信がない。7-2）
} rcjml_ball_t;

static inline rcjml_ball_t rcjml_decode_ball(const float* out, float dist_mu, float dist_sigma) {
  rcjml_ball_t r;
  r.angle_deg = atan2f(out[1], out[0]) * 57.29578f;
  r.distance_cm = expf(out[2] * dist_sigma + dist_mu);
  r.p_hold = 1.0f / (1.0f + expf(-out[3]));
  r.confidence = sqrtf(out[0] * out[0] + out[1] * out[1]);
  return r;
}
