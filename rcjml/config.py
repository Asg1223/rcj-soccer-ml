"""機体の設定ファイル（YAML）を読む。

設定ファイルの場所は、次の順で探す。
    1. load_config() に渡したパス
    2. 環境変数 RCJML_CONFIG
    3. configs/examples/ring12.yaml（教材用の例）

自チームの設定は、チーム用（非公開）リポジトリの configs/robot.yaml に書き、
環境変数 RCJML_CONFIG でその場所を教える。公開リポジトリには置かない。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "examples" / "ring12.yaml"

BALL_TYPES = ("large_ir", "golf_ir")
SENSOR_MCUS = ("arduino", "esp32")


@dataclass
class IRConfig:
    sensor_count: int
    angles_deg: list[float]
    max_reading: int = 1023


@dataclass
class SerialConfig:
    port: str | None = None
    baud: int = 115200


@dataclass
class RobotConfig:
    name: str
    ir: IRConfig
    ball_type: str = "large_ir"
    sensor_mcu: str = "arduino"
    serial: SerialConfig = field(default_factory=SerialConfig)
    data_dir: Path = PACKAGE_ROOT / "data" / "ir-synthetic"
    source: Path | None = None


def _equal_angles(n: int) -> list[float]:
    """正面を 0°、反時計回りを正として、n 個を等間隔に並べた角度。"""
    return [round(360.0 * i / n, 3) for i in range(n)]


def load_config(path: str | os.PathLike | None = None) -> RobotConfig:
    if path is None:
        path = os.environ.get("RCJML_CONFIG") or DEFAULT_CONFIG
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    ir_raw = raw.get("ir", {})
    n = int(ir_raw["sensor_count"])
    angles = ir_raw.get("angles_deg") or _equal_angles(n)
    if len(angles) != n:
        raise ValueError(
            f"{path}: ir.angles_deg の数（{len(angles)}）が sensor_count（{n}）と合いません"
        )
    ir = IRConfig(
        sensor_count=n,
        angles_deg=[float(a) for a in angles],
        max_reading=int(ir_raw.get("max_reading", 1023)),
    )

    ball_type = raw.get("ball", {}).get("type", "large_ir")
    if ball_type not in BALL_TYPES:
        raise ValueError(f"{path}: ball.type は {BALL_TYPES} のどれかにしてください")
    sensor_mcu = raw.get("board", {}).get("sensor_mcu", "arduino")
    if sensor_mcu not in SENSOR_MCUS:
        raise ValueError(f"{path}: board.sensor_mcu は {SENSOR_MCUS} のどれかにしてください")

    serial_raw = raw.get("serial", {}) or {}
    serial = SerialConfig(port=serial_raw.get("port"), baud=int(serial_raw.get("baud", 115200)))

    # 相対パスは「設定ファイルがあるフォルダ」から数える
    data_dir = Path(raw.get("data", {}).get("dir", "../../data/ir-synthetic")).expanduser()
    if not data_dir.is_absolute():
        data_dir = (path.parent / data_dir).resolve()

    return RobotConfig(
        name=str(raw.get("name", path.stem)),
        ir=ir,
        ball_type=ball_type,
        sensor_mcu=sensor_mcu,
        serial=serial,
        data_dir=data_dir,
        source=path,
    )


def example_config(sensor_count: int = 12) -> Path:
    """教材用の例の設定ファイル（configs/examples/ringN.yaml）の場所。"""
    path = PACKAGE_ROOT / "configs" / "examples" / f"ring{sensor_count}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"例の設定ファイルがありません: {path}")
    return path


def describe(cfg: RobotConfig) -> str:
    step = 360.0 / cfg.ir.sensor_count
    lines = [
        f"設定名        : {cfg.name}",
        f"設定ファイル  : {cfg.source}",
        f"IR センサ     : {cfg.ir.sensor_count} 個（等間隔なら {step:.1f}° おき）",
        f"センサ値の上限: {cfg.ir.max_reading}",
        f"ボール        : {cfg.ball_type}",
        f"センサ側      : {cfg.sensor_mcu}",
        f"データの場所  : {cfg.data_dir}",
    ]
    return "\n".join(lines)


def deg_wrap(a: float) -> float:
    """角度を -180〜180° に収める。"""
    return (a + 180.0) % 360.0 - 180.0


__all__ = ["RobotConfig", "IRConfig", "SerialConfig", "load_config", "example_config", "describe", "deg_wrap"]
