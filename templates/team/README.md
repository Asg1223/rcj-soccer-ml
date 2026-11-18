# チーム用リポジトリ（非公開）

このリポジトリは **Private（非公開）** で作ってください。自チームの機体の設定、実機で取ったデータ、画像、学習済みモデルを置きます。

教材のコードは公開リポジトリ [rcj-soccer-ml](https://github.com/Asg1223/rcj-soccer-ml) にあり、このリポジトリからはそれを呼び出して使います。

## 置くもの

| フォルダ | 中身 |
|---|---|
| `configs/robot.yaml` | 自分の機体の設定（センサの個数・角度、ボール、マイコン） |
| `data/ir/` | 実機で取った IR データ（CSV） |
| `data/camera/` | 撮った画像とラベル（知り合いのチームのロボットが写ったものを含む） |
| `models/` | 学習済みモデル、書き出した C ヘッダ |
| `notebooks/` | チーム用の実験ノート |

## 教材のノートブックで自分の設定を使う

環境変数 `RCJML_CONFIG` に、このリポジトリの `configs/robot.yaml` の場所を入れます（くわしくは教材の 0-4）。

**Windows（PowerShell）**

```powershell
$env:RCJML_CONFIG = "C:\Users\あなた\rcj-soccer-ml-team\configs\robot.yaml"
```

**Mac**

```bash
export RCJML_CONFIG=~/rcj-soccer-ml-team/configs/robot.yaml
```

## 注意

- このリポジトリを **Public にしない**
- 知り合いのチームの画像は、撮影と利用の許可をもらったものだけを置く。チームの外に出さない
- 画像が増えて大きくなったら Google Drive に移し、ここには置き場所だけ書く
- Colab で使うときは、データを Google Drive に置いてマウントする（教材の 0-3）
