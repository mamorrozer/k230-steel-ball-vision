#!/usr/bin/env python3
"""Manually export one new 512x288 ONNX without touching historical exports."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from fixed_shape_yolo import command_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, help="Must end with _512x288.onnx")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--device", default="0")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    command_export(SimpleNamespace(**vars(args), imgsz="288,512"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
