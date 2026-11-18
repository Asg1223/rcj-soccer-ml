#!/usr/bin/env python3
"""教材のページ（docs/）から、ノートブック（notebooks/）を作る。

ページの「行頭から始まる ```python のコード」だけを、見出しと一緒にノートブックに入れる。
タブや注意書きの中（字下げされたコード）や、python 以外のコードは入れない。
ページを直したら、これを実行してノートブックも作り直す。

    python tools/build_notebooks.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "Asg1223/rcj-soccer-ml"

NOTEBOOKS = {
    "01_ir_data.ipynb": {
        "title": "第1部 IR データを集める",
        "pages": [
            "part1-ir-data/ball-and-sensor.md",
            "part1-ir-data/send-to-pc.md",
            "part1-ir-data/labels.md",
            "part1-ir-data/explore.md",
        ],
    },
    "02_angle_estimation.ipynb": {
        "title": "第2部 機械学習の基本：角度推定",
        "pages": [
            "part2-ml-basics/baseline.md",
            "part2-ml-basics/linear-regression.md",
            "part2-ml-basics/angle-ring.md",
            "part2-ml-basics/evaluation.md",
            "part2-ml-basics/scikit-learn.md",
        ],
    },
}

SETUP = '''# Colab のときだけ、教材リポジトリを取ってきて読み込めるようにする
import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    !git clone -q --depth 1 https://github.com/{repo}.git /content/rcj-soccer-ml
    sys.path.insert(0, "/content/rcj-soccer-ml")
print("Colab で実行中" if IN_COLAB else "ローカルで実行中")'''


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def page_cells(rel: str) -> list[dict]:
    """1ページ分のセル。H1 と、コードの直前の見出しを markdown として入れる。"""
    text = (ROOT / "docs" / rel).read_text(encoding="utf-8")
    url = f"https://{REPO.split('/')[0].lower()}.github.io/{REPO.split('/')[1]}/{rel[:-3]}/"
    cells: list[dict] = []
    last_heading = None
    pending_heading = None
    in_code = False
    buf: list[str] = []
    for line in text.splitlines():
        if in_code:
            if line.startswith("```"):
                in_code = False
                if pending_heading:
                    cells.append(md(pending_heading))
                    pending_heading = None
                cells.append(code("\n".join(buf)))
                buf = []
            else:
                buf.append(line)
            continue
        if line.startswith("```python"):
            in_code = True
            continue
        m = re.match(r"^(#{1,3}) (.+)$", line)
        if m:
            level, title = len(m.group(1)), m.group(2)
            if level == 1:
                cells.append(md(f"## {title}\n\n説明は教材のページを見てください：{url}"))
            else:
                if title != last_heading:
                    pending_heading = f"{'#' * (level + 1)} {title}"
                    last_heading = title
    return cells


def build(name: str, spec: dict) -> None:
    badge = (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
             f"(https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{name})")
    cells = [
        md(f"# {spec['title']}\n\n{badge}\n\n"
           "このノートブックは、教材のページのコードを順番に並べたものです（`tools/build_notebooks.py` で自動生成）。\n"
           "上のセルから順に `Shift + Enter` で実行してください。"),
        code(SETUP.format(repo=REPO)),
    ]
    for rel in spec["pages"]:
        cells += page_cells(rel)
    for i, c in enumerate(cells):
        c["id"] = f"c{i:03d}"
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / "notebooks" / name
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n_code = sum(c["cell_type"] == "code" for c in cells)
    print(f"{name}: {len(cells)} セル（コード {n_code}）")


if __name__ == "__main__":
    for name, spec in NOTEBOOKS.items():
        build(name, spec)
