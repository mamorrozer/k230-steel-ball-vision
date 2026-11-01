#!/usr/bin/env python3
"""Find legacy 320-size references without treating archived experiments as active code."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

TEXT_SUFFIXES = {".py", ".ps1", ".sh", ".yaml", ".yml", ".json", ".toml", ".md", ".txt", ".cfg", ".ini"}
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules"}
HISTORY_DIR_PREFIXES = ("runs", "predict", "archive", "backup", "history")
HISTORY_DIRS = {"datasets", "reports", "tests", "weights", "det_results", "mp_deployment_source"}
LEGACY_RE = re.compile(r"(?<!\d)320(?!\d)")
DEPLOYMENT_KEYS = ("model_input_size", "input_width", "input_height", "image_size", "imgsz", "config_preprocess")


@dataclass(frozen=True)
class SizeMatch:
    path: Path
    line: int
    category: str
    text: str

    def to_json(self) -> dict[str, object]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


def is_history_path(path: Path) -> bool:
    lower_parts = [part.lower() for part in path.parts]
    if any(part in SKIP_DIRS or part in HISTORY_DIRS for part in lower_parts):
        return True
    if any(part.startswith(HISTORY_DIR_PREFIXES) for part in lower_parts):
        return True
    if any(left == right for left, right in zip(lower_parts, lower_parts[1:])):
        return True
    name = path.name.lower()
    return "_320" in name or "320x320" in name or "320_trained" in name


def classify_match(path: Path, text: str) -> str:
    if is_history_path(path):
        return "history"
    lowered = text.lower()
    return "active" if any(key in lowered for key in DEPLOYMENT_KEYS) else "history"


def iter_text_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() in TEXT_SUFFIXES:
            yield root
        return
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part.lower() in SKIP_DIRS for part in path.parts):
            continue
        if path.name == Path(__file__).name:
            continue
        yield path


def scan_paths(paths: Iterable[Path], include_history: bool = False) -> list[SizeMatch]:
    matches: list[SizeMatch] = []
    for root in (Path(item).resolve() for item in paths):
        if not root.exists():
            raise FileNotFoundError(root)
        for path in iter_text_files(root):
            try:
                if path.stat().st_size > 4 * 1024 * 1024:
                    continue
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, text in enumerate(lines, 1):
                if not LEGACY_RE.search(text):
                    continue
                category = classify_match(path, text)
                if category == "history" and not include_history:
                    continue
                matches.append(SizeMatch(path, line_number, category, text.strip()))
    order = {"history": 0, "active": 1}
    return sorted(matches, key=lambda item: (order[item.category], str(item.path).lower(), item.line))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Roots to scan; default is the project root")
    parser.add_argument("--include-history", action="store_true", help="Also list archived 320 experiments/results")
    parser.add_argument("--fail-on-active-320", action="store_true", help="Return exit code 2 if active 320 references remain")
    parser.add_argument("--json", type=Path, help="Optional machine-readable report path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    roots = args.paths or [project_root]
    matches = scan_paths(roots, include_history=args.include_history)
    active = [item for item in matches if item.category == "active"]
    history = [item for item in matches if item.category == "history"]
    print(f"INPUT_SIZE_SCAN,active_320={len(active)},history_320={len(history)}")
    for item in matches:
        print(f"{item.category.upper()}|{item.path}:{item.line}|{item.text}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"roots": [str(Path(root).resolve()) for root in roots], "active_320_count": len(active), "history_320_count": len(history), "matches": [item.to_json() for item in matches]}
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.fail_on_active_320 and active:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
