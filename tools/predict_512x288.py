#!/usr/bin/env python3
"""Run one non-overwriting 512x288 prediction job."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from fixed_shape_yolo import command_predict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    # command_predict uses imgsz=(288,512), save=True and exist_ok=False.
    command_predict(SimpleNamespace(**vars(args), imgsz="288,512"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
