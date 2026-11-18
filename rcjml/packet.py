"""マイコン同士の通信のパケット（教材 5-2）。firmware/common/rcjml_packet.h の Python 版。

    0xA5 0x5A  TYPE  LEN  SEQ  PAYLOAD  CRC8
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

SYNC = b"\xA5\x5A"
MAX_PAYLOAD = 64

T_IR_RAW = 0x01
T_BALL = 0x02
T_MOTOR = 0x10
T_MODE = 0x11


def crc8(data: bytes) -> int:
    """CRC-8（多項式 0x07、初期値 0）。C 版と同じ計算。"""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def encode(ptype: int, seq: int, payload: bytes) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"中身が長すぎます（{len(payload)} > {MAX_PAYLOAD}）")
    body = bytes([ptype, len(payload), seq & 0xFF]) + payload
    return SYNC + body + bytes([crc8(body)])


@dataclass
class Packet:
    type: int
    seq: int
    payload: bytes


class Parser:
    """バイトを少しずつ渡すと、そろったパケットを返す。C の rcjml_parser_t と同じ動き。"""

    def __init__(self):
        self.state = 0
        self.buf = bytearray()
        self.ok = self.crc_errors = self.len_errors = 0

    def feed(self, data: bytes) -> list[Packet]:
        out = []
        for c in data:
            if self.state == 0:
                if c == 0xA5:
                    self.state = 1
            elif self.state == 1:
                if c == 0x5A:
                    self.state, self.buf = 2, bytearray()
                else:
                    self.state = 1 if c == 0xA5 else 0
            elif self.state == 2:
                self.buf.append(c)
                if len(self.buf) == 2 and self.buf[1] > MAX_PAYLOAD:
                    self.len_errors += 1
                    self.state = 1 if c == 0xA5 else 0
                elif len(self.buf) >= 3 and len(self.buf) == 3 + self.buf[1]:
                    self.state = 3
            else:
                self.state = 0
                if crc8(bytes(self.buf)) != c:
                    self.crc_errors += 1
                    continue
                self.ok += 1
                out.append(Packet(self.buf[0], self.buf[2], bytes(self.buf[3:])))
        return out


def pack_ir(values) -> bytes:
    return struct.pack(f"<{len(values)}H", *[int(v) for v in values])


def unpack_ir(payload: bytes) -> list[int]:
    return list(struct.unpack(f"<{len(payload) // 2}H", payload))


def pack_ball(angle_deg: float, distance_cm: float, p_hold: float, confidence: float) -> bytes:
    def clip(v, lo, hi):
        return max(lo, min(hi, v))

    return struct.pack(
        "<hHBB",
        int(round(clip(angle_deg * 10, -32768, 32767))),
        int(round(clip(distance_cm * 10, 0, 65535))),
        int(round(clip(p_hold * 255, 0, 255))),
        int(round(clip(confidence * 255, 0, 255))),
    )


def unpack_ball(payload: bytes) -> dict:
    a, d, h, c = struct.unpack("<hHBB", payload)
    return {"angle_deg": a / 10, "distance_cm": d / 10, "p_hold": h / 255, "confidence": c / 255}


def pack_motor(m1: int, m2: int, m3: int, m4: int) -> bytes:
    return struct.pack("<4h", *[max(-1000, min(1000, int(v))) for v in (m1, m2, m3, m4)])


def send_sim(port: str, angle: float | None = None, distance: float = 40.0, difficulty: str = "normal",
             rate_hz: float = 50.0, seconds: float = 30.0) -> None:
    """シミュレータのセンサ値を IR_RAW パケットにして送る（実機のセンサ側の代わり。5-3）。

    angle を省略すると、ボールがロボットのまわりをゆっくり回る。
    """
    import time

    import numpy as np
    import serial

    from rcjml import load_config
    from rcjml.sim import sample_session, simulate

    cfg = load_config()
    rng = np.random.default_rng(0)
    sess = sample_session(difficulty, rng)
    with serial.Serial(port, 115200, timeout=0) as ser:
        time.sleep(2.0)
        t0 = time.monotonic()
        seq = 0
        while time.monotonic() - t0 < seconds:
            t = time.monotonic() - t0
            a = angle if angle is not None else ((t * 30.0 + 180.0) % 360.0) - 180.0
            v = simulate(cfg, a, distance, difficulty=difficulty, session=sess, rng=rng)[0]
            ser.write(encode(T_IR_RAW, seq, pack_ir(v)))
            seq = (seq + 1) & 0xFF
            if seq % 25 == 0:
                print(f"送信中：ボール {a:+6.1f}°, {distance:.0f} cm")
            time.sleep(1.0 / rate_hz)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(prog="python -m rcjml.packet")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send-sim", help="シミュレータの値をパケットで送る")
    s.add_argument("--port", required=True)
    s.add_argument("--angle", type=float, help="省略するとボールが回る")
    s.add_argument("--distance", type=float, default=40.0)
    s.add_argument("--difficulty", default="normal")
    s.add_argument("--seconds", type=float, default=30.0)
    a = p.parse_args()
    send_sim(a.port, a.angle, a.distance, a.difficulty, seconds=a.seconds)
