# RCJ Soccer 機械学習入門

ロボカップジュニア サッカー（赤外線）向けの機械学習教材です。
IR センサとカメラ（M5Stack CoreS3）でボール・ゴール・相手ロボットを見つけるモデルを作り、マイコンで動かすところまで進みます。

- 教材サイト: https://asg1223.github.io/rcj-soccer-ml/
- 理論 3 : 実装 7。本文は中高生向け、「もっと深く」で大学生レベルまで
- Colab とローカル（VS Code + Jupyter）の両方で動く

## フォルダ

| フォルダ | 中身 |
|---|---|
| `docs/` | 教材サイトの本文（Markdown） |
| `notebooks/` | Colab とローカル共通のノートブック |
| `rcjml/` | 教材用 Python パッケージ（設定ファイル、IR シミュレータ など） |
| `firmware/` | Arduino / ESP32 / CoreS3 のスケッチ |
| `configs/examples/` | 設定ファイルの例（特定の機体の値ではない） |
| `data/ir-synthetic/` | シミュレータで作った合成データ |
| `data/camera-sample/` | 公開用のサンプル画像 |
| `templates/team/` | チーム用（非公開）リポジトリのひな形 |
| `tools/` | 公開チェックなどの道具 |

## サイトを手元で見る

```bash
pip install -r requirements-docs.txt
zensical serve
```

ブラウザで http://localhost:8000/rcj-soccer-ml/ を開きます。

## 公開しないもの

このリポジトリは公開です。自チームの機体の設定、実機のデータ、画像、学習済みモデルは置きません（チーム用の非公開リポジトリに置きます）。
コミット前のチェックを有効にしてから作業してください。

```bash
pip install pre-commit
pre-commit install
```

くわしくは教材の「0-3 GitHub とチーム用リポジトリ」を参照。

## ライセンス

[MIT License](LICENSE)。教材の本文・コード・合成データ・サンプル画像のすべてに適用します。
自由に使い、改変し、再配布できます。再配布するときは、著作権表示とライセンス文を残してください。
