"""学習したモデルを、マイコンで使える C のヘッダに書き出す（教材 4-1, 4-3, 4-4）。

    from rcjml.export import layers_from_keras, write_mlp_header
    layers = layers_from_keras(model)
    write_mlp_header(layers, "ballnet_model.h", name="ballnet")

書き出したヘッダは firmware/common/rcjml_mlp.h と一緒に使う。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rcjml.nn import DIST_MU, DIST_SIGMA


@dataclass
class Dense:
    W: np.ndarray        # (入力, 出力)。Keras と同じ向き
    b: np.ndarray        # (出力,)
    activation: str      # "relu" か "linear"


def layers_from_keras(model) -> list[Dense]:
    """Keras の Sequential（Dense だけ）から重みを取り出す。"""
    out = []
    for layer in model.layers:
        if not hasattr(layer, "kernel"):
            continue
        act = layer.get_config().get("activation", "linear")
        if act not in ("relu", "linear"):
            raise ValueError(f"{layer.name}: 活性化関数 {act} は書き出せません（relu か linear にする）")
        W, b = layer.get_weights()
        out.append(Dense(W.astype(np.float32), b.astype(np.float32), act))
    return out


def forward(layers: list[Dense], X: np.ndarray, quantized: bool = False) -> np.ndarray:
    """NumPy で同じ計算をする（C の結果の答え合わせ用）。"""
    h = np.asarray(X, dtype=np.float32)
    for L in layers:
        W = dequantize(*quantize(L.W)) if quantized else L.W
        h = h @ W + L.b
        if L.activation == "relu":
            h = np.maximum(h, 0.0)
    return h


def quantize(W: np.ndarray) -> tuple[np.ndarray, float]:
    """重みを int8（-127〜127）にする。層ごとに 1 つの倍率（scale）を使う。"""
    scale = float(np.abs(W).max()) / 127.0 or 1.0
    q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * np.float32(scale)


def _fmt(values, per_line: int = 8, fmt: str = "{:.8e}f") -> str:
    items = [fmt.format(v) for v in np.ravel(values)]
    lines = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return ",\n  ".join(lines)


def write_mlp_header(
    layers: list[Dense],
    path: str | Path,
    *,
    name: str = "ballnet",
    quantized: bool = False,
    input_scale: float = 1.0 / 1023.0,
    note: str = "",
) -> Path:
    """C のヘッダを書き出す。quantized=True なら重みを int8 にする（4-3）。"""
    path = Path(path)
    N = name.upper()
    n_in = layers[0].W.shape[0]
    n_out = layers[-1].W.shape[1]
    width = max(L.W.shape[1] for L in layers)
    n_params = sum(L.W.size + L.b.size for L in layers)
    lines = [
        f"// {path.name} — rcjml.export で自動生成。手で書きかえない",
        "// !!! 自分のデータで学習したモデルは、チーム用（非公開）リポジトリに置く",
    ]
    if note:
        lines.append(f"// {note}")
    lines += [
        f"// 層: {' → '.join([str(n_in)] + [str(L.W.shape[1]) for L in layers])}"
        f"（パラメータ {n_params} 個、重み {'int8' if quantized else 'float'}）",
        "#pragma once",
        '#include "rcjml_mlp.h"',
        "",
        f"#define {N}_N_IN {n_in}",
        f"#define {N}_N_OUT {n_out}",
        f"#define {N}_N_LAYERS {len(layers)}",
        f"#define {N}_MAX_WIDTH {width}",
        f"static const float {N}_INPUT_SCALE = {input_scale:.10e}f;   // センサ値にかけて 0〜1 にする",
        f"static const float {N}_DIST_MU = {DIST_MU:.6f}f;",
        f"static const float {N}_DIST_SIGMA = {DIST_SIGMA:.6f}f;",
        "",
    ]
    entries = []
    for k, L in enumerate(layers):
        Wt = L.W.T  # C では「出力 × 入力」の順
        if quantized:
            q, scale = quantize(L.W)
            lines.append(f"static const int8_t {name}_w{k}[{Wt.size}] RCJML_PROGMEM = {{\n  "
                         + _fmt(q.T, per_line=16, fmt="{:d}") + "\n};")
            w_ptr, wq_ptr, sc = "0", f"{name}_w{k}", f"{scale:.10e}f"
        else:
            lines.append(f"static const float {name}_w{k}[{Wt.size}] RCJML_PROGMEM = {{\n  " + _fmt(Wt) + "\n};")
            w_ptr, wq_ptr, sc = f"{name}_w{k}", "0", "1.0f"
        lines.append(f"static const float {name}_b{k}[{L.b.size}] RCJML_PROGMEM = {{\n  " + _fmt(L.b) + "\n};")
        act = "RCJML_RELU" if L.activation == "relu" else "RCJML_LINEAR"
        entries.append(f"  {{{L.W.shape[0]}, {L.W.shape[1]}, {w_ptr}, {wq_ptr}, {sc}, {name}_b{k}, {act}}}")
    lines += [
        "",
        f"static const rcjml_layer_t {name}_layers[{N}_N_LAYERS] = {{",
        ",\n".join(entries),
        "};",
        "",
        f"// センサの値（0〜1 にそろえたもの）から出力を計算する",
        f"static inline void {name}_forward(const float* in, float* out) {{",
        f"  static float a[{N}_MAX_WIDTH], b[{N}_MAX_WIDTH];",
        f"  rcjml_forward({name}_layers, {N}_N_LAYERS, in, out, a, b);",
        "}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_to_sketch(model_or_layers, sketch_dir, *, quantized: bool = False, allow_public: bool = False,
                     input_scale: float = 1.0 / 1023.0, note: str = "") -> list[Path]:
    """スケッチのフォルダに、モデルのヘッダ（ballnet_model.h）と rcjml_mlp.h を置く（4-1）。

    自分のデータで学習したモデルはチームの情報なので、公開リポジトリの中には書き出さない。
    シミュレータのデータで練習するときだけ allow_public=True にする（.gitignore で除外される）。
    """
    import shutil

    from rcjml.config import PACKAGE_ROOT

    sketch_dir = Path(sketch_dir).expanduser().resolve()
    root = PACKAGE_ROOT.resolve()
    if not allow_public and (sketch_dir == root or root in sketch_dir.parents):
        raise SystemExit(
            f"{sketch_dir} は公開リポジトリ（教材）の中です。\n"
            "スケッチのフォルダをチーム用リポジトリにコピーしてから、そこに書き出してください（4-1）。\n"
            "シミュレータのデータで練習するだけなら allow_public=True にします（コミットはされません）。"
        )
    layers = model_or_layers if isinstance(model_or_layers, list) else layers_from_keras(model_or_layers)
    sketch_dir.mkdir(parents=True, exist_ok=True)
    header = write_mlp_header(layers, sketch_dir / "ballnet_model.h", name="ballnet",
                              quantized=quantized, input_scale=input_scale, note=note)
    return [header, *copy_common_headers(sketch_dir)]


COMMON_HEADERS = ("rcjml_mlp.h", "rcjml_packet.h")


def copy_common_headers(sketch_dir) -> list[Path]:
    """firmware/common の共通ヘッダを、スケッチのフォルダにコピーする。

    Arduino IDE は、スケッチのフォルダの外にあるヘッダを読めないため。
    ターミナルからは  python -m rcjml.export headers <スケッチのフォルダ>
    """
    import shutil

    sketch_dir = Path(sketch_dir).expanduser().resolve()
    common = Path(__file__).resolve().parent.parent / "firmware" / "common"
    out = []
    for name in COMMON_HEADERS:
        shutil.copy(common / name, sketch_dir / name)
        out.append(sketch_dir / name)
    return out


def tflite_to_c(tflite_bytes: bytes, path: str | Path, name: str = "ballnet_tflite") -> Path:
    """.tflite のモデルを C の配列にする（4-4。xxd -i と同じ）。"""
    path = Path(path)
    body = ",\n  ".join(
        ", ".join(f"0x{b:02x}" for b in tflite_bytes[i:i + 16]) for i in range(0, len(tflite_bytes), 16)
    )
    path.write_text(
        f"// {path.name} — rcjml.export.tflite_to_c で自動生成\n"
        "// !!! 自分のデータで学習したモデルは、チーム用（非公開）リポジトリに置く\n"
        "#pragma once\n"
        f"alignas(16) const unsigned char {name}[] = {{\n  {body}\n}};\n"
        f"const unsigned int {name}_len = {len(tflite_bytes)};\n",
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "headers":
        for f in copy_common_headers(sys.argv[2]):
            print("コピーしました:", f)
    else:
        print("使い方: python -m rcjml.export headers <スケッチのフォルダ>")
