// 0-2 環境チェック用：Arduino / ESP32 共通
// 1秒ごとにシリアルモニタへ数字を送る。LED があるボードでは点滅もする。
// シリアルモニタの速度は 115200 にする。

const unsigned long BAUD = 115200;
unsigned long count = 0;

void setup() {
  Serial.begin(BAUD);
#ifdef LED_BUILTIN
  pinMode(LED_BUILTIN, OUTPUT);
#endif
}

void loop() {
  Serial.print("hello RCJ Soccer ML  count=");
  Serial.println(count++);
#ifdef LED_BUILTIN
  digitalWrite(LED_BUILTIN, count % 2);
#endif
  delay(1000);
}
