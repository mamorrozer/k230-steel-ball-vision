#!/usr/bin/env python3
"""Read-only validation for single-class steel-ball YOLO datasets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ("train", "val", "test")


def absolute_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def resolve_config_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return absolute_path(path if path.is_absolute() else base / path)


def replace_images_with_labels(path: Path) -> Path | None:
    parts = list(path.parts)
    indexes = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not indexes:
        return None
    parts[indexes[-1]] = "labels"
    return Path(*parts)


def expected_label_path(image_path: Path) -> Path | None:
    label_path = replace_images_with_labels(image_path)
    return label_path.with_suffix(".txt") if label_path else None


def iter_image_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield absolute_path(path)


def resolve_manifest_image(entry: str, dataset_root: Path, manifest_path: Path) -> Path:
    candidate = Path(entry.strip()).expanduser()
    if candidate.is_absolute():
        return absolute_path(candidate)
    root_candidate = absolute_path(dataset_root / candidate)
    if root_candidate.exists():
        return root_candidate
    return absolute_path(manifest_path.parent / candidate)


def collect_images(source: Path, dataset_root: Path) -> tuple[list[Path], list[str], list[Path]]:
    errors: list[str] = []
    label_roots: list[Path] = []
    if not source.exists():
        return [], [f"数据源不存在: {source}"], []
    if source.is_dir():
        label_root = replace_images_with_labels(source)
        if label_root:
            label_roots.append(label_root)
        images = list(iter_image_files(source))
        if not images:
            errors.append(f"数据源中没有支持的图片: {source}")
        return images, errors, label_roots
    if source.suffix.lower() in IMAGE_SUFFIXES:
        return [absolute_path(source)], errors, []
    if source.suffix.lower() != ".txt":
        return [], [f"不支持的数据源类型: {source}"], []

    images: list[Path] = []
    try:
        lines = source.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        return [], [f"无法读取图片清单 {source}: {exc}"], []
    for line_number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        image_path = resolve_manifest_image(text, dataset_root, source)
        if not image_path.exists():
            errors.append(f"图片清单第 {line_number} 行不存在: {image_path}")
        elif image_path.suffix.lower() not in IMAGE_SUFFIXES:
            errors.append(f"图片清单第 {line_number} 行不是支持的图片: {image_path}")
        else:
            images.append(image_path)
    if not images and not errors:
        errors.append(f"图片清单为空: {source}")
    return images, errors, label_roots


def check_image(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
        return None
    except (OSError, SyntaxError, ValueError, UnidentifiedImageError) as exc:
        return str(exc)


def parse_class_id(token: str) -> int:
    if any(character in token for character in ".eE"):
        raise ValueError("类别编号必须是整数")
    return int(token, 10)


def check_label(path: Path) -> tuple[bool, int, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        return False, 0, [{"path": str(path), "line": 0, "message": f"无法读取标签: {exc}", "text": ""}]

    nonempty_lines = [(number, line.strip()) for number, line in enumerate(text.splitlines(), 1) if line.strip()]
    if not nonempty_lines:
        return True, 0, issues

    valid_instances = 0
    for line_number, line in nonempty_lines:
        fields = line.split()
        issue = {"path": str(path), "line": line_number, "message": "", "text": line}
        if len(fields) != 5:
            issue["message"] = f"标签必须恰好为 5 列，实际为 {len(fields)} 列"
            issues.append(issue)
            continue
        try:
            class_id = parse_class_id(fields[0])
        except ValueError:
            issue["message"] = "类别编号必须是整数 0"
            issues.append(issue)
            continue
        if class_id != 0:
            issue["message"] = f"钢球项目类别编号只能为 0，实际为 {class_id}"
            issues.append(issue)
            continue
        try:
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            issue["message"] = "坐标必须是数值"
            issues.append(issue)
            continue
        if not all(math.isfinite(value) for value in coordinates):
            issue["message"] = "坐标必须是有限数值"
            issues.append(issue)
            continue
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            issue["message"] = "x_center、y_center、width、height 必须在 0 到 1 之间"
            issues.append(issue)
            continue
        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            issue["message"] = "width 和 height 必须大于 0"
            issues.append(issue)
            continue
        valid_instances += 1
    return False, valid_instances, issues


def configured_class_ids(names: Any) -> list[int] | None:
    if isinstance(names, list):
        return list(range(len(names)))
    if isinstance(names, dict):
        try:
            return sorted(int(key) for key in names)
        except (TypeError, ValueError):
            return None
    return None


def validate_split(name: str, values: list[Any], dataset_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sources": [],
        "image_count": 0,
        "label_count": 0,
        "empty_label_count": 0,
        "instance_count": 0,
        "missing_labels": [],
        "corrupt_images": [],
        "invalid_labels": [],
        "extra_labels": [],
        "errors": [],
    }
    images: list[Path] = []
    label_roots: list[Path] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            result["errors"].append(f"{name} 包含无效路径值: {value!r}")
            continue
        source = resolve_config_path(value, dataset_root)
        result["sources"].append(str(source))
        source_images, source_errors, source_label_roots = collect_images(source, dataset_root)
        images.extend(source_images)
        result["errors"].extend(source_errors)
        label_roots.extend(source_label_roots)

    unique_images = sorted(set(images), key=lambda item: str(item).lower())
    result["image_count"] = len(unique_images)
    expected_labels: set[Path] = set()
    for image_path in unique_images:
        image_error = check_image(image_path)
        if image_error:
            result["corrupt_images"].append({"path": str(image_path), "message": image_error})
        label_path = expected_label_path(image_path)
        if label_path is None:
            result["errors"].append(f"图片路径中没有 images 目录，无法确定标签位置: {image_path}")
            continue
        label_path = absolute_path(label_path)
        expected_labels.add(label_path)
        if not label_path.exists():
            result["missing_labels"].append(str(label_path))
            continue
        result["label_count"] += 1
        is_empty, instances, issues = check_label(label_path)
        if is_empty:
            result["empty_label_count"] += 1
        result["instance_count"] += instances
        result["invalid_labels"].extend(issues)

    actual_labels: set[Path] = set()
    for label_root in set(label_roots):
        if label_root.exists() and label_root.is_dir():
            actual_labels.update(absolute_path(path) for path in label_root.rglob("*.txt") if path.is_file())
    result["extra_labels"] = [str(path) for path in sorted(actual_labels - expected_labels, key=lambda item: str(item).lower())]
    return result


def normalize_split_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def validate_dataset(data_yaml: Path) -> dict[str, Any]:
    data_yaml = absolute_path(data_yaml)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml 不存在: {data_yaml}")
    with data_yaml.open("r", encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("data.yaml 顶层必须是映射")

    root_value = config.get("path")
    dataset_root = resolve_config_path(str(root_value), data_yaml.parent) if root_value else data_yaml.parent
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_yaml": str(data_yaml),
        "dataset_root": str(dataset_root),
        "configured_splits": [],
        "training_allowed": False,
        "blocking_error_count": 0,
        "warning_count": 0,
        "splits": {},
        "errors": [],
        "warnings": [],
    }
    if not dataset_root.is_dir():
        report["errors"].append(f"数据集根目录不存在: {dataset_root}")
    class_ids = configured_class_ids(config.get("names"))
    if class_ids != [0]:
        report["errors"].append(f"钢球项目 names 只能定义类别 0，实际为: {class_ids}")

    for split_name in SPLIT_NAMES:
        if split_name not in config:
            if split_name in ("train", "val"):
                report["errors"].append(f"data.yaml 缺少必需的 {split_name} 配置")
            continue
        report["configured_splits"].append(split_name)
        split_result = validate_split(split_name, normalize_split_value(config[split_name]), dataset_root)
        report["splits"][split_name] = split_result

        report["errors"].extend(f"{split_name}: {message}" for message in split_result["errors"])
        report["errors"].extend(f"{split_name}: 缺少标签 {path}" for path in split_result["missing_labels"])
        report["errors"].extend(
            f"{split_name}: 损坏图片 {item['path']}: {item['message']}" for item in split_result["corrupt_images"]
        )
        report["errors"].extend(
            f"{split_name}: 非法标签 {item['path']}:{item['line']}: {item['message']}"
            for item in split_result["invalid_labels"]
        )
        report["warnings"].extend(f"{split_name}: 多余标签 {path}" for path in split_result["extra_labels"])

    report["blocking_error_count"] = len(report["errors"])
    report["warning_count"] = len(report["warnings"])
    report["training_allowed"] = report["blocking_error_count"] == 0
    return report


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict[str, Any]) -> str:
    status = "允许训练" if report["training_allowed"] else "阻止训练"
    lines = [
        "# YOLO 数据集检查报告",
        "",
        f"- 结论：**{status}**",
        f"- `data.yaml`：`{report['data_yaml']}`",
        f"- 数据集根目录：`{report['dataset_root']}`",
        f"- 阻断问题：{report['blocking_error_count']}",
        f"- 警告：{report['warning_count']}",
        "",
        "## 数据统计",
        "",
        "| Split | 图片 | 标签 | 空标签 | 目标实例 | 缺失标签 | 损坏图片 | 非法标签 | 多余标签 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in report["configured_splits"]:
        split = report["splits"][split_name]
        lines.append(
            f"| {split_name} | {split['image_count']} | {split['label_count']} | "
            f"{split['empty_label_count']} | {split['instance_count']} | {len(split['missing_labels'])} | "
            f"{len(split['corrupt_images'])} | {len(split['invalid_labels'])} | {len(split['extra_labels'])} |"
        )

    lines.extend(["", "## 阻断问题", ""])
    lines.extend(f"- {markdown_escape(message)}" for message in report["errors"])
    if not report["errors"]:
        lines.append("- 无")
    lines.extend(["", "## 警告", ""])
    lines.extend(f"- {markdown_escape(message)}" for message in report["warnings"])
    if not report["warnings"]:
        lines.append("- 无")
    lines.extend(
        [
            "",
            "## 规则说明",
            "",
            "- 0 字节或仅含空白的同名 `txt` 是合法负样本。",
            "- 缺少同名 `txt` 不会被自动视为负样本，而是直接阻止训练。",
            "- 本工具只读检查，不会修改图片、标签或 `data.yaml`。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读检查钢球 YOLO 数据集")
    parser.add_argument("data_yaml", type=Path, help="Ultralytics data.yaml 路径")
    parser.add_argument("--markdown", type=Path, default=Path.cwd() / "yolo_dataset_validation.md")
    parser.add_argument("--json", dest="json_path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = validate_dataset(args.data_yaml)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"VALIDATION_ERROR: {exc}", file=sys.stderr)
        return 2

    markdown_path = absolute_path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    if args.json_path:
        json_path = absolute_path(args.json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"YOLO_DATASET_VALIDATION,allowed={int(report['training_allowed'])},"
        f"errors={report['blocking_error_count']},warnings={report['warning_count']},"
        f"markdown={markdown_path}"
    )
    return 0 if report["training_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
