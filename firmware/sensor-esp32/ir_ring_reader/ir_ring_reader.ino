// IR センサリングの読み取り（ESP32 版）  教材 1-2
//
// 各センサの出力が LOW（= ボールの光を受けている）だった時間の割合を測り、
// 0〜MAX_READING の値にしてシリアルに送る。
//
// 送るデータ（1行 = 1回の測定）:
//   IR,<時刻ms>,<s0>,<s1>,...,<sN-1>
// 起動したときに、設定を知らせる行を1回送る:
//   # RCJML IR N=<センサ数> MAX=<上限> WINDOW_US=<測る時間>
//
// !!! 下の「設定」は例です。自分の機体の値（ピン・個数）に書きかえたものは、
// !!! チーム用（非公開）リポジトリにコピーして保存してください。

// ===== 設定 =====
// センサをつないだピン。s0（正面）から反時計回りの順に並べる
// ESP32 では、起動に使うピン（0, 2, 12, 15 など）や入力専用ピンに注意してボードの説明書を確認する
const uint8_t SENSOR_PINS[] = {4, 5, 13, 14, 16, 17, 18, 19, 21, 22, 23, 25};
// 1回の測定で見る時間（マイクロ秒）。ボールの光は 833µs で1周期なので、その数倍にする
const unsigned long WINDOW_US = 2500;
// 値の上限（設定ファイルの ir.max_reading と合わせる）
const int MAX_READING = 1023;
// 受光モジュールの出力が「受けているとき LOW」なら true
const bool ACTIVE_LOW = true;
const unsigned long BAUD = 115200;
// ===============

const uint8_t N = sizeof(SENSOR_PINS) / sizeof(SENSOR_PINS[0]);
uint32_t lowCount[N];

void setup() {
  Serial.begin(BAUD);
  for (uint8_t i = 0; i < N; i++) {
    pinMode(SENSOR_PINS[i], INPUT_PULLUP);
  }
  delay(200);
  Serial.print("# RCJML IR N=");
  Serial.print(N);
  Serial.print(" MAX=");
  Serial.print(MAX_READING);
  Serial.print(" WINDOW_US=");
  Serial.println(WINDOW_US);
}

void loop() {
  for (uint8_t i = 0; i < N; i++) lowCount[i] = 0;
  uint32_t sweeps = 0;

  // WINDOW_US のあいだ、全センサをくり返し見て「受けている」回数を数える
  unsigned long start = micros();
  while (micros() - start < WINDOW_US) {
    for (uint8_t i = 0; i < N; i++) {
      bool on = (digitalRead(SENSOR_PINS[i]) == LOW) == ACTIVE_LOW;
      if (on) lowCount[i]++;
    }
    sweeps++;
  }

  // 割合を 0〜MAX_READING に直して送る
  Serial.print("IR,");
  Serial.print(millis());
  for (uint8_t i = 0; i < N; i++) {
    long v = (long)lowCount[i] * MAX_READING / sweeps;
    Serial.print(',');
    Serial.print(v);
  }
  Serial.println();
}
