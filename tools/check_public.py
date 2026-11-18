#!/usr/bin/env python3
"""公開リポジトリに「公開してはいけないもの」が混ざっていないか調べる。

使い方:
    python tools/check_public.py FILE...   # 指定したファイルだけ調べる（pre-commit から呼ばれる）
    python tools/check_public.py --all     # git で管理しているファイルを全部調べる（CI で使う）

調べること:
    1. 置いてはいけない場所にあるファイル（実データ・チームの設定・学習済みモデル）
    2. 決められた場所以外にある画像と CSV
    3. 大きすぎるファイル
    4. 出力（グラフや表）が残ったままのノートブック
    5. .private-words に書いた言葉（自チームのセンサの型番など）を含むファイル

.private-words はリポジトリのいちばん上に自分で作る。1行に1語。
このファイル自体は .gitignore で除外されているので公開されない。
"""

from __future__ import annotations

import fnmatch
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 1_000_000  # 1 MB

# 1. このパターンに当てはまるファイルは公開リポジトリに置かない
FORBIDDEN = [
    "data/real/*",
    "data/ir/*",
    "data/camera/*",
    "models/*",
    "configs/robot*.yaml",
    "configs/team/*",
    "*.tflite",
    "*.h5",
    "*.keras",
    "*.onnx",
    "*.espdl",
    "*.eim",
]

# 2. 画像と CSV はここにだけ置いてよい
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
IMAGE_OK = ["docs/*", "data/camera-sample/*"]
CSV_EXT = {".csv", ".tsv"}
CSV_OK = ["data/ir-synthetic/*", "docs/*"]

# 3. サイズ上限の例外
LARGE_OK: list[str] = []

TEXT_EXT = {
    ".md", ".py", ".ipynb", ".yaml", ".yml", ".toml", ".txt", ".ino", ".cpp",
    ".c", ".h", ".hpp", ".js", ".css", ".html", ".json", ".csv", ".tsv",
}


def match(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def load_private_words() -> list[str]:
    f = ROOT / ".private-words"
    if not f.exists():
        return []
    words = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.append(line)
    return words


def notebook_has_outputs(p: Path) -> bool:
    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code" and cell.get("outputs"):
            return True
    return False


def check(rel: str, words: list[str]) -> list[str]:
    problems = []
    p = ROOT / rel
    if not p.is_file():
        return problems
    ext = p.suffix.lower()

    if match(rel, FORBIDDEN):
        problems.append("公開リポジトリに置かない種類のファイルです（チーム用リポジトリへ）")
    if ext in IMAGE_EXT and not match(rel, IMAGE_OK):
        problems.append("画像は docs/ か data/camera-sample/ にだけ置けます")
    if ext in CSV_EXT and not match(rel, CSV_OK):
        problems.append("CSV は data/ir-synthetic/ か docs/ にだけ置けます")
    if p.stat().st_size > MAX_BYTES and not match(rel, LARGE_OK):
        problems.append(f"ファイルが大きすぎます（{p.stat().st_size // 1000} KB > {MAX_BYTES // 1000} KB）")
    if ext == ".ipynb" and notebook_has_outputs(p):
        problems.append("ノートブックに出力が残っています（nbstripout で消してからコミット）")
    if words and ext in TEXT_EXT:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = ""
        low = text.lower()
        for w in words:
            if w.lower() in low:
                problems.append(f".private-words の言葉「{w}」が含まれています")
    return problems


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


def main(argv: list[str]) -> int:
    if argv == ["--all"]:
        files = tracked_files()
    else:
        files = [Path(a).resolve().relative_to(ROOT).as_posix() for a in argv]

    words = load_private_words()
    bad = 0
    for rel in files:
        for msg in check(rel, words):
            print(f"NG  {rel}: {msg}")
            bad += 1

    if bad:
        print(f"\n{bad} 件の問題があります。コミット（公開）を止めました。")
        print("本当に公開してよいものなら、tools/check_public.py の許可リストを見直してください。")
        return 1
    print(f"OK  {len(files)} ファイルを確認しました。公開して問題になりそうなものはありません。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
