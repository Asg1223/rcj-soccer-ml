# 0-4 設定ファイル

## ねらい

- 機体ごとに違う情報を **設定ファイル（YAML）** に書ける
- 教材のコードを書きかえずに、自分の機体の設定に切り替えられる

## 理論：コードと設定を分ける

センサの個数、並べた角度、ボールの種類……。こうした「機体ごとに違う値」をコードの中に直接書くと、2つ困ることがあります。

1. 機体が変わるたびに、コードのあちこちを書きかえることになる
2. コードを公開すると、チームの情報まで一緒に公開されてしまう

そこで、**値は設定ファイルに、やり方はコードに** 分けます。教材のコードは設定ファイルを読んで動くので、設定を差し替えるだけで自分の機体に合わせられます。

## 設定ファイルの中身

教材用の例は `configs/examples/` にあります（IR センサ 8 / 12 / 16 個の3つ）。どれも例で、特定のチームの値ではありません。

```yaml
name: example-ring12

ir:
  sensor_count: 12      # センサの個数
  angles_deg: null      # 並べた角度。null なら等間隔
  max_reading: 1023     # センサ値の上限

ball:
  type: large_ir        # large_ir（大型）または golf_ir（ゴルフボールサイズ）

board:
  sensor_mcu: arduino   # arduino または esp32

serial:
  port: null            # 例：COM3、/dev/cu.usbmodem1101
  baud: 115200

data:
  dir: ../../data/ir-synthetic   # データの場所（この設定ファイルから見た位置）
```

### 角度の決め方

ロボットを真上から見て、**正面を 0°**、**反時計回り（左回り）を ＋** とします。

```text
              0°（正面）
               ↑
     +90°  ←  ロボット  →  -90°（= 270°）
               ↓
             180°（後ろ）
```

センサが等間隔に並んでいて、1つ目（`s0`）が正面を向いているなら `angles_deg: null` のままで構いません。
そうでなければ、`s0` から順に角度を並べます（例：`[0, 25, 50, ...]`）。

!!! private "自分の機体の値はチーム用リポジトリへ"
    自分の機体の設定は、チーム用（非公開）リポジトリの `configs/robot.yaml` に書きます。
    公開リポジトリの `configs/` に `robot.yaml` を置くと、`.gitignore` と pre-commit で止められます。

---

## 手を動かす 1：自分の設定ファイルを書く

チーム用リポジトリの `configs/robot.yaml` を開き、`TODO` と書かれたところを自分の機体の値に書きかえます。
まだ機体が決まっていなければ、例のまま（12 個・等間隔）で構いません。

## 手を動かす 2：教材のコードに自分の設定を使わせる

教材のコードは、設定ファイルを次の順で探します。

1. `load_config("ファイルの場所")` と直接指定したもの
2. 環境変数 `RCJML_CONFIG` に入っている場所
3. どちらもなければ、例の `configs/examples/ring12.yaml`

いちばん手軽なのは **環境変数** です。一度設定すれば、どのノートブックでも自分の設定が使われます。

=== "Colab"

    ```python
    from google.colab import drive
    drive.mount("/content/drive")

    import os
    os.environ["RCJML_CONFIG"] = "/content/drive/MyDrive/rcj-soccer-ml-team/configs/robot.yaml"
    ```

=== "ローカル（Windows）"

    PowerShell で、VS Code を起動する前に設定します。

    ```powershell
    $env:RCJML_CONFIG = "$HOME\rcj-soccer-ml-team\configs\robot.yaml"
    code ~\rcj-soccer-ml
    ```

=== "ローカル（Mac）"

    ターミナルで、VS Code を起動する前に設定します。

    ```bash
    export RCJML_CONFIG=~/rcj-soccer-ml-team/configs/robot.yaml
    code ~/rcj-soccer-ml
    ```

    毎回打つのが面倒なら、`~/.zshrc` の最後に `export` の行を書いておきます。

読み込めたかどうかは、ノートブックで確かめます。

```python
from rcjml import load_config, describe

cfg = load_config()
print(describe(cfg))
```

`設定ファイル` の行に、自分の `robot.yaml` の場所が出ていれば成功です。

## 手を動かす 3：センサの個数でどう変わるか見てみる

例の設定を3つ読み込んで、同じ場所にボールを置いたときのセンサ値を比べます。

```python
import numpy as np
import matplotlib.pyplot as plt
from rcjml import load_config, example_config
from rcjml.sim import simulate

fig, axes = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={"projection": "polar"})
for ax, n in zip(axes, [8, 12, 16]):
    cfg = load_config(example_config(n))
    values = simulate(cfg, angle_deg=45, distance_cm=40, difficulty="easy",
                      rng=np.random.default_rng(0))[0]
    theta = np.deg2rad(cfg.ir.angles_deg)
    ax.set_theta_zero_location("N")
    ax.bar(theta, values, width=2 * np.pi / n * 0.8, alpha=0.7)
    ax.plot([np.deg2rad(45)] * 2, [0, cfg.ir.max_reading], color="tab:orange", lw=3)
    ax.set_ylim(0, cfg.ir.max_reading)
    ax.set_title(f"{n} sensors")
plt.show()
```

`example_config(n)` は、教材に入っている例の設定ファイル（センサ n 個）の場所を返す関数です。

## チェック問題

1. ロボットの真左にあるボールは何度でしょう。
2. センサが等間隔ではないとき、設定ファイルのどこを書きかえますか。
3. 環境変数 `RCJML_CONFIG` を設定し忘れると、どの設定が使われますか。

??? note "答え"
    1. +90° です（正面 0°、反時計回りが ＋）。
    2. `ir.angles_deg` に、`s0` から順に角度を並べます。
    3. 例の `configs/examples/ring12.yaml` が使われます。`describe(cfg)` の表示で必ず確かめましょう。

## つまずきポイント

??? failure "`ir.angles_deg の数が sensor_count と合いません` と出る"
    角度のリストの個数と `sensor_count` が違います。どちらかを直します。

??? failure "YAML の読み込みでエラーになる"
    YAML は **字下げ（スペースの数）** で構造を表します。タブ文字を使っていないか、同じ階層の行の先頭がそろっているかを確かめます。

## もっと深く

!!! deeper "設定を環境変数で渡す理由"
    設定ファイルの場所をコードに書くと、人によって PC のフォルダ構成が違うので、そのたびにコードを直すことになります。
    環境変数で渡せば、**コードは誰の PC でも同じ**、**違いは環境だけ** にできます。Web サービスの開発で広く使われている考え方（The Twelve-Factor App の「設定」）と同じです。
    また、秘密の情報（ここではチームの機体情報）をコードの外に置くことは、パスワードや API キーの扱いと同じ発想です。
