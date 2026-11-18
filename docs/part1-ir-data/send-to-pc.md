# 1-3 PC に送る

## ねらい

- マイコンから届いた行を、Python で数字のリストにできる
- 教材のロガー（`rcjml.logger`）で、センサの値を見たり保存したりできる
- 実機がなくても、シミュレータを「にせのマイコン」にして同じ操作を練習できる

---

## 理論：シリアル通信で何が届いているか

マイコンは USB ケーブルを通して、**1行ずつの文字** を送っています。

```text
IR,1538,412,388,96,0,0,0,0,0,0,0,51,302
```

PC 側でやることは3つだけです。

1. 1行受け取る（改行まで読む）
2. 形を確かめる（`IR,` で始まるか、数の個数が合っているか）
3. カンマで区切って数字にする

形を確かめるのは、通信の最初や途中で **こわれた行** が混ざることがあるからです（接続した直後の読みかけの行、ノイズで文字化けした行など）。こわれた行は捨てます。

---

## 手を動かす 1：1行を数字にする関数を書く

実機がなくても、文字の行を作って試せます。

```python
def parse_line(line, n):
    """'IR,時刻,s0,s1,...' の行をセンサ値のリストにする。形が違えば None。"""
    line = line.strip()
    if not line.startswith("IR,"):
        return None
    parts = line.split(",")
    if len(parts) != n + 2:          # "IR" と時刻の 2 つ + センサ n 個
        return None
    try:
        return [int(p) for p in parts[2:]]
    except ValueError:
        return None

tests = [
    "IR,1538,412,388,96,0,0,0,0,0,0,0,51,302",   # 正しい行
    "# RCJML IR N=12 MAX=1023 WINDOW_US=2500",   # お知らせの行
    "IR,1540,412,388,96,0,0",                    # 途中で切れた行
    "IR,1541,41?,388,96,0,0,0,0,0,0,0,51,302",   # 文字化けした行
]
for t in tests:
    print(parse_line(t, 12))
```

1行目だけがリストになり、残りは `None` になれば正解です。教材のロガーの中でも、これと同じ関数（`rcjml.logger.parse_line`）を使っています。

## 手を動かす 2：pyserial で直接読む（実機）

実機をつないでいるなら、`pyserial` で直接読んでみます。ポート名は自分の環境に合わせます。

=== "Windows"

    ```python
    import serial
    ser = serial.Serial("COM3", 115200, timeout=1)
    for _ in range(10):
        print(ser.readline().decode("ascii", errors="replace").strip())
    ser.close()
    ```

    ポート名は Arduino IDE の「ツール → ポート」か、デバイスマネージャーの「ポート（COM と LPT）」で確かめます。

=== "Mac"

    ```python
    import serial
    ser = serial.Serial("/dev/cu.usbmodem1101", 115200, timeout=1)
    for _ in range(10):
        print(ser.readline().decode("ascii", errors="replace").strip())
    ser.close()
    ```

    ポート名はターミナルで `ls /dev/cu.*` と打つと一覧が出ます。

!!! warning "Arduino IDE のシリアルモニタを閉じる"
    シリアルポートは、同時に1つのソフトしか使えません。Python から読むときは、Arduino IDE のシリアルモニタを閉じておきます。

## 手を動かす 3：教材のロガーを使う

毎回コードを書かなくてよいように、教材にはロガーを用意しています。ターミナル（仮想環境に入った状態）で使います。

### 値を見る（`monitor`）

=== "実機"

    ```bash
    python -m rcjml.logger monitor
    ```

    ポートは自動で探します。見つからないときは `--port COM3` のように指定するか、設定ファイルの `serial.port` に書きます。

=== "シミュレータ"

    ```bash
    python -m rcjml.logger monitor --port sim --angle 60 --distance 30
    ```

    `--port sim` にすると、シミュレータが「にせのマイコン」になります。

センサごとの値が棒グラフで表示され、どんどん書きかわります。`Ctrl + C` で終わります。

### 1か所だけ記録する（`record`）

```bash
python -m rcjml.logger record --session 1 --angle 30 --distance 40
```

ボールを「角度 30°、距離 40 cm」に置いた状態で、20 回分の値を記録します。保存先は、設定ファイルの `data.dir` にある `ir_log.csv` です。

!!! private "保存先はチーム用リポジトリ"
    ロガーは、**教材用の例の設定のまま** だったり、**保存先が公開リポジトリの中** だったりすると、保存せずに止まります。実機のデータを公開しないための安全装置です。
    0-4 の手順で、環境変数 `RCJML_CONFIG` にチーム用リポジトリの `configs/robot.yaml` を指定してから使ってください。

    シミュレータで練習するだけなら、`--allow-public` を付けると `data/practice/` に保存できます。このフォルダは公開されません（`.gitignore` で除外済み）。

    ```bash
    python -m rcjml.logger record --session 1 --angle 30 --distance 40 --port sim --allow-public
    ```

保存される CSV はこんな形です。

```text
s0,s1,...,s11,angle_deg,distance_cm,has_ball,session,time
650,650,...,244,30.0,40.0,0,2,2026-11-18T19:19:45
```

たくさんの場所を順番に記録する方法（`plan`）は、次の 1-4 で使います。

---

## チェック問題

1. `parse_line` で、センサの個数 `n` を確かめないと、どんな困ったことが起きますか。
2. ロガーが「例の設定のままです」と言って止まりました。どうすればよいですか。
3. Python からポートを開こうとしたら「アクセスが拒否されました」と出ました。まず何を確かめますか。

??? note "答え"
    1. 途中で切れた行や、スケッチと設定ファイルでセンサの個数が違うときに、まちがった列にずれた値が入ってしまいます。
    2. チーム用リポジトリの `configs/robot.yaml` を環境変数 `RCJML_CONFIG` で指定します（0-4）。練習だけなら `--allow-public` を付けます。
    3. Arduino IDE のシリアルモニタなど、ほかのソフトがポートを使っていないかを確かめます。

## つまずきポイント

??? failure "`TimeoutError: ... 行そろいませんでした` と出る"
    正しい形の行が届いていません。次の順で確かめます。

    1. Arduino IDE のシリアルモニタで、`IR,` の行が流れているか
    2. スケッチのセンサの個数と、設定ファイルの `ir.sensor_count` が同じか
    3. 速度（115200）がスケッチと設定ファイルで同じか

??? failure "`AttributeError: module 'serial' has no attribute 'Serial'`"
    `pyserial` とは別の `serial` というパッケージが入っています。0-2 のつまずきポイントの手順で直します。

??? failure "Windows で `monitor` の表示がくずれる"
    古いコマンドプロンプトだと、画面の書きかえがうまく表示されないことがあります。Windows Terminal か VS Code のターミナルで実行します。

## もっと深く

!!! deeper "文字で送るか、バイナリで送るか"
    この教材では、人が読んで確かめやすいように **文字（カンマ区切り）** で送っています。ただし文字は、同じ数を送るのにバイト数が多くなります（`1023,` は 5 バイト、バイナリなら 2 バイト）。
    第5部でマイコン同士をつなぐときは、速さと確実さのために **バイナリのパケット**（決まった先頭の印、データ、チェックサム）にします。そこで、なぜ先頭の印やチェックサムが要るのかを、今回の「こわれた行」の話とつなげて説明します。
