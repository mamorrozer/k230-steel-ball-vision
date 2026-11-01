#!/usr/bin/env python3
"""Fixed HxW Ultralytics detection train/val/predict/export entry point.

Ultralytics 8.4.104 accepts two-dimensional sizes for predict/export but
coerces train/val to a scalar. This module keeps the installed environment
unchanged and supplies a detection Dataset/Trainer/Validator whose actual
batch tensors are fixed to the requested (height, width).
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from copy import copy
from pathlib import Path
from typing import Any, Sequence

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import (
    Albumentations,
    Compose,
    Format,
    LetterBox,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
)
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model


DEFAULT_FIXED_SHAPE = (288, 512)  # (height, width)
DEFAULT_CAMERA_SHAPE = (720, 1280)  # (height, width)


def parse_imgsz(value: str | Sequence[int]) -> tuple[int, int]:
    """Parse Ultralytics-style H,W input without guessing its ordering."""
    if isinstance(value, str):
        parts = [part.strip() for part in value.lower().replace("x", ",").split(",") if part.strip()]
    else:
        parts = [str(part) for part in value]
    if len(parts) != 2:
        raise ValueError("imgsz must contain exactly height,width, for example 288,512")
    shape = (int(parts[0]), int(parts[1]))
    return validate_fixed_shape(shape)


def validate_fixed_shape(shape: Sequence[int], stride: int = 32) -> tuple[int, int]:
    """Validate a landscape (height, width) shape aligned to the model stride."""
    if len(shape) != 2:
        raise ValueError("fixed shape must be (height, width)")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if width < height:
        raise ValueError("expected landscape height,width ordering; width must be >= height")
    if height % stride or width % stride:
        raise ValueError(f"height and width must both be multiples of stride {stride}")
    return height, width


def build_size_manifest(
    model_shape_hw: Sequence[int] = DEFAULT_FIXED_SHAPE,
    camera_shape_hw: Sequence[int] = DEFAULT_CAMERA_SHAPE,
) -> dict[str, Any]:
    """Return explicit orderings used by every deployment stage."""
    height, width = validate_fixed_shape(model_shape_hw)
    camera_height, camera_width = int(camera_shape_hw[0]), int(camera_shape_hw[1])
    return {
        "ultralytics_imgsz_hw": [height, width],
        "onnx_nchw": [1, 3, height, width],
        "nncase_input_width": width,
        "nncase_input_height": height,
        "k230_model_input_size_wh": [width, height],
        "k230_rgb888p_size_wh": [camera_width, camera_height],
    }


def make_test_hyp():
    """Return the installed Ultralytics defaults for dataset-level tests."""
    return get_cfg()


class FixedShapeYOLODataset(YOLODataset):
    """YOLO detection dataset that always emits one exact HxW tensor shape."""

    def __init__(self, *args, fixed_shape: Sequence[int] = DEFAULT_FIXED_SHAPE, **kwargs):
        self.fixed_shape = validate_fixed_shape(fixed_shape)
        kwargs["rect"] = False
        kwargs["imgsz"] = max(self.fixed_shape)
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp=None) -> Compose:
        height, width = self.fixed_shape
        if hyp is None:
            hyp = get_cfg()

        if self.augment:
            # The installed Mosaic/MixUp/CutMix path is square-only. Disable it
            # explicitly and retain fixed-shape geometric and color augmentation.
            hyp.mosaic = 0.0
            hyp.mixup = 0.0
            hyp.cutmix = 0.0
            hyp.copy_paste = 0.0
            transforms = Compose(
                [
                    LetterBox(new_shape=(height, width), auto=False, scaleup=True, stride=self.stride),
                    RandomPerspective(
                        degrees=hyp.degrees,
                        translate=hyp.translate,
                        scale=hyp.scale,
                        shear=hyp.shear,
                        perspective=hyp.perspective,
                        size=(width, height),
                    ),
                    Albumentations(p=1.0, transforms=getattr(hyp, "augmentations", None)),
                    RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
                    RandomFlip(direction="vertical", p=hyp.flipud, flip_idx=[]),
                    RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=[]),
                ]
            )
        else:
            transforms = Compose(
                [LetterBox(new_shape=(height, width), auto=False, scaleup=False, stride=self.stride)]
            )

        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,
            )
        )
        return transforms


def build_fixed_shape_dataset(
    cfg,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    mode: str,
    stride: int,
    fixed_shape: Sequence[int],
) -> FixedShapeYOLODataset:
    """Build a detection-only dataset without altering data.yaml or labels."""
    if cfg.task != "detect":
        raise ValueError("fixed_shape_yolo.py currently supports task=detect only")
    return FixedShapeYOLODataset(
        img_path=img_path,
        imgsz=max(fixed_shape),
        fixed_shape=fixed_shape,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=False,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


class FixedShapeDetectionValidator(DetectionValidator):
    fixed_shape = DEFAULT_FIXED_SHAPE

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        return build_fixed_shape_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode,
            self.stride,
            self.fixed_shape,
        )

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess(batch)
        actual = tuple(int(v) for v in batch["img"].shape[-2:])
        if actual != tuple(self.fixed_shape):
            raise RuntimeError(f"validator batch shape {actual} != fixed shape {self.fixed_shape}")
        return batch


class FixedShapeDetectionTrainer(DetectionTrainer):
    fixed_shape = DEFAULT_FIXED_SHAPE

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_fixed_shape_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode,
            stride,
            self.fixed_shape,
        )

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess_batch(batch)
        actual = tuple(int(v) for v in batch["img"].shape[-2:])
        if actual != tuple(self.fixed_shape):
            raise RuntimeError(f"training batch shape {actual} != fixed shape {self.fixed_shape}")
        return batch

    def get_validator(self):
        FixedShapeDetectionValidator.fixed_shape = tuple(self.fixed_shape)
        return FixedShapeDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )


def _write_json(path_value: str | None, payload: dict[str, Any]) -> None:
    if not path_value:
        return
    output = Path(path_value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _metrics_payload(metrics, shape: tuple[int, int], split: str) -> dict[str, Any]:
    values = getattr(metrics, "results_dict", {})
    return {
        "input_size_hw": list(shape),
        "split": split,
        "precision": float(values.get("metrics/precision(B)", 0.0)),
        "recall": float(values.get("metrics/recall(B)", 0.0)),
        "map50": float(values.get("metrics/mAP50(B)", 0.0)),
        "map50_95": float(values.get("metrics/mAP50-95(B)", 0.0)),
    }


def command_train(args) -> None:
    shape = parse_imgsz(args.imgsz)
    FixedShapeDetectionTrainer.fixed_shape = shape
    run_dir = Path(args.project) / args.name
    if run_dir.exists():
        raise FileExistsError(f"training output already exists: {run_dir}")
    model = YOLO(args.model)
    model.train(
        trainer=FixedShapeDetectionTrainer,
        data=args.data,
        imgsz=max(shape),
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        exist_ok=False,
        rect=False,
        multi_scale=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        amp=not args.no_amp,
    )


def command_val(args) -> None:
    shape = parse_imgsz(args.imgsz)
    FixedShapeDetectionValidator.fixed_shape = shape
    output_dir = Path(args.project) / args.name
    if output_dir.exists():
        raise FileExistsError(f"validation output already exists: {output_dir}")
    model = YOLO(args.model)
    metrics = model.val(
        validator=FixedShapeDetectionValidator,
        data=args.data,
        imgsz=max(shape),
        batch=args.batch,
        split=args.split,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=False,
        rect=False,
    )
    _write_json(args.metrics_json, _metrics_payload(metrics, shape, args.split))


def command_predict(args) -> None:
    shape = parse_imgsz(args.imgsz)
    output_dir = Path(args.project) / args.name
    if output_dir.exists():
        raise FileExistsError(f"prediction output already exists: {output_dir}")
    model = YOLO(args.model)
    model.predict(
        source=args.source,
        imgsz=shape,
        conf=args.conf,
        device=args.device,
        project=args.project,
        name=args.name,
        rect=False,
        save=True,
        exist_ok=False,
    )


def command_export(args) -> None:
    shape = parse_imgsz(args.imgsz)
    output = Path(args.output).resolve()
    expected_suffix = f"_{shape[1]}x{shape[0]}"
    if output.suffix.lower() != ".onnx" or not output.stem.endswith(expected_suffix):
        raise ValueError(f"ONNX output must end with {expected_suffix}.onnx")
    if output.exists():
        raise FileExistsError(f"ONNX output already exists: {output}")
    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="yolo_export_512x288_") as temp_dir:
        staged_pt = Path(temp_dir) / f"{output.stem}.pt"
        shutil.copy2(model_path, staged_pt)
        exported = YOLO(str(staged_pt)).export(
            format="onnx",
            imgsz=shape,
            batch=1,
            dynamic=False,
            simplify=True,
            opset=args.opset,
            device=args.device,
        )
        shutil.move(str(Path(exported)), output)
    _write_json(str(output.with_suffix(".size.json")), build_size_manifest(shape))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--model", required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--imgsz", default="288,512", help="height,width")
    train.add_argument("--epochs", type=int, required=True)
    train.add_argument("--batch", type=int, required=True)
    train.add_argument("--device", default="0")
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--project", required=True)
    train.add_argument("--name", required=True)
    train.add_argument("--no-amp", action="store_true", help="Disable AMP and its online self-check.")
    train.set_defaults(handler=command_train)

    val = subparsers.add_parser("val")
    val.add_argument("--model", required=True)
    val.add_argument("--data", required=True)
    val.add_argument("--imgsz", default="288,512", help="height,width")
    val.add_argument("--batch", type=int, required=True)
    val.add_argument("--split", choices=("val", "test"), default="val")
    val.add_argument("--device", default="0")
    val.add_argument("--workers", type=int, default=0)
    val.add_argument("--project", required=True)
    val.add_argument("--name", required=True)
    val.add_argument("--metrics-json")
    val.set_defaults(handler=command_val)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--model", required=True)
    predict.add_argument("--source", required=True)
    predict.add_argument("--imgsz", default="288,512", help="height,width")
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--device", default="0")
    predict.add_argument("--project", required=True)
    predict.add_argument("--name", required=True)
    predict.set_defaults(handler=command_predict)

    export = subparsers.add_parser("export")
    export.add_argument("--model", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--imgsz", default="288,512", help="height,width")
    export.add_argument("--opset", type=int, default=12)
    export.add_argument("--device", default="0")
    export.set_defaults(handler=command_export)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
