"""IR センサリングのデータを PC に集める（教材 1-3、1-4）。

使い方（ターミナルで、仮想環境に入ってから）:

    # センサの値を画面で見る
    python -m rcjml.logger monitor

    # ボールを 30°・40 cm に置いて、20 回分を記録する
    python -m rcjml.logger record --angle 30 --distance 40 --session 1

    # 決めた位置を順番に案内してもらいながら記録する（いちばん楽）
    python -m rcjml.logger plan --session 1

    # 実機がないとき：シミュレータを「にせのマイコン」として使う
    python -m rcjml.logger plan --session 1 --port sim

記録したデータは、設定ファイルの data.dir にある ir_log.csv に追記される。
**教材用の例の設定や、公開リポジトリの中には保存しない**（チームのデータを守るため）。
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rcjml.config import PACKAGE_ROOT, RobotConfig, load_config

LOG_NAME = "ir_log.csv"


# ---------------------------------------------------------------------------
# 1行を読む
# ---------------------------------------------------------------------------


def parse_line(line: str, n: int) -> list[int] | None:
    """マイコンから届いた1行を、センサ値のリストにする。形が違う行は None。

    >>> parse_line("IR,1234,10,20,30", 3)
    [10, 20, 30]
    """
    line = line.strip()
    if not line.startswith("IR,"):
        return None
    parts = line.split(",")
    if len(parts) != n + 2:
        return None
    try:
        return [int(p) for p in parts[2:]]
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# マイコン（本物、またはシミュレータのにせもの）
# ---------------------------------------------------------------------------


def find_port() -> str:
    """マイコンらしいシリアルポートを探す。見つからなければエラー。"""
    from serial.tools import list_ports

    ports = [p for p in list_ports.comports() if p.vid is not None]
    if len(ports) == 1:
        return ports[0].device
    names = ", ".join(p.device for p in ports) or "なし"
    raise RuntimeError(
        f"マイコンのポートを1つに決められません（見つかったポート: {names}）。"
        " --port で指定するか、設定ファイルの serial.port に書いてください。"
    )


class SimulatedBoard:
    """シミュレータを使った「にせのマイコン」。実機と同じ形の行を返す。"""

    def __init__(self, cfg: RobotConfig, difficulty: str = "normal", seed: int = 0):
        from rcjml.sim import sample_session

        self.cfg = cfg
        self.difficulty = difficulty
        self.rng = np.random.default_rng(seed)
        self.session = sample_session(difficulty, self.rng)
        self.angle = 0.0
        self.distance = 50.0
        self.t0 = time.monotonic()

    def place_ball(self, angle_deg: float, distance_cm: float) -> None:
        self.angle, self.distance = angle_deg, distance_cm

    def readline(self) -> str:
        from rcjml.sim import simulate

        time.sleep(0.005)
        v = simulate(
            self.cfg, self.angle, self.distance, difficulty=self.difficulty,
            session=self.session, rng=self.rng,
        )[0]
        ms = int((time.monotonic() - self.t0) * 1000)
        return "IR," + str(ms) + "," + ",".join(str(x) for x in v)

    def close(self) -> None:
        pass


class SerialBoard:
    """本物のマイコン（USB シリアル）。"""

    def __init__(self, port: str, baud: int):
        import serial

        if "://" in port:  # テスト用（loop:// など）
            self.ser = serial.serial_for_url(port, baudrate=baud, timeout=1.0)
        else:
            self.ser = serial.Serial(port, baud, timeout=1.0)
            time.sleep(2.0)  # Arduino は接続するとリセットされるので少し待つ
            self.ser.reset_input_buffer()

    def place_ball(self, angle_deg: float, distance_cm: float) -> None:
        pass  # 本物のボールは人が置く

    def readline(self) -> str:
        return self.ser.readline().decode("ascii", errors="replace")

    def close(self) -> None:
        self.ser.close()


def open_board(cfg: RobotConfig, port: str | None, difficulty: str = "normal"):
    port = port or cfg.serial.port
    if port == "sim":
        return SimulatedBoard(cfg, difficulty)
    return SerialBoard(port or find_port(), cfg.serial.baud)


def read_samples(board, n_sensors: int, count: int, timeout_s: float = 10.0) -> np.ndarray:
    """形の正しい行を count 個集める。"""
    rows: list[list[int]] = []
    bad = 0
    t_end = time.monotonic() + timeout_s
    while len(rows) < count:
        if time.monotonic() > t_end:
            raise TimeoutError(
                f"{timeout_s:.0f} 秒待っても {count} 行そろいませんでした（正しい行 {len(rows)}、形が違う行 {bad}）。"
                " センサの個数がスケッチと設定ファイルで合っているか確かめてください。"
            )
        v = parse_line(board.readline(), n_sensors)
        if v is None:
            bad += 1
        else:
            rows.append(v)
    return np.asarray(rows)


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------


PRACTICE_DIR = PACKAGE_ROOT / "data" / "practice"   # 練習用。.gitignore で除外している


def output_path(cfg: RobotConfig, out: str | None, allow_public: bool) -> Path:
    """保存先を決める。公開リポジトリの中や、例の設定のままでは保存させない。"""
    if out:
        path = Path(out).expanduser()
    elif allow_public:
        path = PRACTICE_DIR / LOG_NAME
    else:
        path = cfg.data_dir / LOG_NAME
    check_output_dir(cfg, path.parent, allow_public)
    return path


def check_output_dir(cfg: RobotConfig, out_dir: Path, allow_public: bool = False) -> None:
    examples = (PACKAGE_ROOT / "configs" / "examples").resolve()
    out = out_dir.resolve()
    if allow_public:
        practice = PRACTICE_DIR.resolve()
        if out != practice and practice not in out.parents:
            raise SystemExit(f"--allow-public のときは {practice} にだけ保存できます。")
        return
    if cfg.source and examples in cfg.source.resolve().parents:
        raise SystemExit(
            "教材用の例の設定（configs/examples）のままです。実機のデータは公開しないので、\n"
            "チーム用リポジトリの configs/robot.yaml を環境変数 RCJML_CONFIG で指定してください（0-4）。\n"
            "シミュレータで練習するだけなら --allow-public を付けると、data/practice/（公開されない）に保存できます。"
        )
    if out == PACKAGE_ROOT.resolve() or PACKAGE_ROOT.resolve() in out.parents:
        raise SystemExit(
            f"保存先 {out} が公開リポジトリ（教材）の中です。\n"
            "設定ファイルの data.dir を、チーム用リポジトリの data/ir などにしてください。"
        )


def append_rows(path: Path, cfg: RobotConfig, values: np.ndarray, *, angle: float,
                distance: float, has_ball: int, session: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    header = [f"s{i}" for i in range(cfg.ir.sensor_count)] + [
        "angle_deg", "distance_cm", "has_ball", "session", "time",
    ]
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        for row in values:
            w.writerow(list(row) + [angle, distance, has_ball, session, stamp])


# ---------------------------------------------------------------------------
# 収集の計画
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spot:
    angle: float
    distance: float
    has_ball: int = 0


def make_plan(
    angles=range(-165, 181, 15),
    distances=(10, 20, 30, 45, 60, 80, 100, 130),
    holds: int = 3,
    seed: int | None = None,
) -> list[Spot]:
    """ボールを置く位置の一覧。毎回並び順を変えて、時間による変化が偏らないようにする。"""
    spots = [Spot(float(a), float(d)) for a in angles for d in distances]
    spots += [Spot(0.0, 3.0, 1) for _ in range(holds)]   # 保持しているとき
    random.Random(seed).shuffle(spots)
    return spots


# ---------------------------------------------------------------------------
# コマンド
# ---------------------------------------------------------------------------


def bar(v: int, vmax: int, width: int = 30) -> str:
    k = int(round(width * v / max(vmax, 1)))
    return "#" * k + "." * (width - k)


def cmd_monitor(args, cfg: RobotConfig) -> None:
    board = open_board(cfg, args.port, args.difficulty)
    if isinstance(board, SimulatedBoard):
        board.place_ball(args.angle or 30.0, args.distance or 40.0)
    if os.name == "nt":
        os.system("")  # Windows のターミナルで画面の書きかえ（エスケープシーケンス）を有効にする
    print("Ctrl + C で終わります")
    try:
        while True:
            v = read_samples(board, cfg.ir.sensor_count, 1)[0]
            lines = [f"s{i:<2d} {a:7.1f}° {x:5d} {bar(x, cfg.ir.max_reading)}"
                     for i, (a, x) in enumerate(zip(cfg.ir.angles_deg, v))]
            print("\033[H\033[J" + "\n".join(lines), flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        board.close()


def cmd_record(args, cfg: RobotConfig) -> None:
    out = output_path(cfg, args.out, args.allow_public)
    board = open_board(cfg, args.port, args.difficulty)
    try:
        board.place_ball(args.angle, args.distance)
        v = read_samples(board, cfg.ir.sensor_count, args.samples)
        append_rows(out, cfg, v, angle=args.angle, distance=args.distance,
                    has_ball=int(args.has_ball), session=args.session)
        print(f"{len(v)} 行を {out} に保存しました（平均: {np.round(v.mean(0)).astype(int).tolist()}）")
    finally:
        board.close()


def cmd_plan(args, cfg: RobotConfig) -> None:
    out = output_path(cfg, args.out, args.allow_public)
    plan = make_plan(seed=args.session)
    if args.limit:
        plan = plan[: args.limit]
    board = open_board(cfg, args.port, args.difficulty)
    auto = isinstance(board, SimulatedBoard)
    print(f"取り直し {args.session}：{len(plan)} か所 × {args.samples} 回。保存先 {out}")
    print("ボールを言われた場所に置いて Enter。s で飛ばす、q で終わる。")
    try:
        for k, spot in enumerate(plan, 1):
            what = "ボールを保持させる" if spot.has_ball else f"角度 {spot.angle:+.0f}°、距離 {spot.distance:.0f} cm"
            if auto:
                print(f"[{k}/{len(plan)}] {what}（シミュレータ）")
            else:
                ans = input(f"[{k}/{len(plan)}] {what} に置いて Enter > ").strip().lower()
                if ans == "q":
                    break
                if ans == "s":
                    continue
            board.place_ball(spot.angle, spot.distance)
            v = read_samples(board, cfg.ir.sensor_count, args.samples)
            append_rows(out, cfg, v, angle=spot.angle, distance=spot.distance,
                        has_ball=spot.has_ball, session=args.session)
    finally:
        board.close()
    print("終わりました。")


def main(argv: list[str] | None = None) -> None:
    # 共通のオプション（コマンドの前でも後ろでも書ける）
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="設定ファイル（省略すると RCJML_CONFIG）")
    common.add_argument("--port", help="シリアルポート（例 COM3、/dev/cu.usbmodem1101）。sim でシミュレータ")
    common.add_argument("--difficulty", default="normal", help="--port sim のときの難しさ")

    p = argparse.ArgumentParser(prog="python -m rcjml.logger", description="IR センサのデータを集める")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("monitor", parents=[common], help="センサの値を画面で見る")
    m.add_argument("--angle", type=float, help="（sim のみ）ボールの角度")
    m.add_argument("--distance", type=float, help="（sim のみ）ボールの距離")

    for name, help_ in [("record", "1か所だけ記録する"), ("plan", "決めた位置を順番に記録する")]:
        s = sub.add_parser(name, parents=[common], help=help_)
        s.add_argument("--session", type=int, required=True, help="何回目の取り直しか（1, 2, 3, …）")
        s.add_argument("--samples", type=int, default=20, help="1か所で記録する回数")
        s.add_argument("--out", help="保存するファイル（省略すると data.dir/ir_log.csv）")
        s.add_argument("--allow-public", action="store_true",
                       help="練習用：例の設定のまま data/practice/（公開されない）に保存する")
        if name == "record":
            s.add_argument("--angle", type=float, required=True, help="ボールの角度（正面 0°、左が +）")
            s.add_argument("--distance", type=float, required=True, help="ボールまでの距離 cm")
            s.add_argument("--has-ball", action="store_true", help="ボールを保持している")
        else:
            s.add_argument("--limit", type=int, help="最初の何か所だけやるか（練習用）")

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    {"monitor": cmd_monitor, "record": cmd_record, "plan": cmd_plan}[args.cmd](args, cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
