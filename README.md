# K230 Steel-Ball Vision Pipeline

**K230 camera deployment, fixed-shape YOLO experiments and control-side position preparation**

This repository presents the vision subsystem of a larger embedded control project. It focuses on the part that can be reviewed independently:

```text
camera → fixed-shape YOLO inference → target selection → pipe-axis projection
       → position / velocity smoothing → diagnostic UART line
```

The main control candidate is a full-frame **352 × 192** KModel. A measured **about 28 FPS** local-LCD run made it the first candidate for control integration. A **512 × 288** model remains available as the accuracy/debug reference at about **20 FPS** on the same K230 test setup.

## Why this is separate from the contest showcase

The contest showcase describes the complete MSPM0 vehicle and system integration. This repository goes deeper into the K230 vision module: fixed input geometry, model comparison, board-side cadence, target selection, and the boundary between a reliable vision output and a future motor loop. It does not claim the motor-control closed loop is complete.

## What is included

| Area | Contents |
| --- | --- |
| Board deployment | CanMV-style K230 integration and diagnostic scripts; vendor runtime files stay outside the repository |
| Geometry | Fixed H×W parsing, explicit camera/model size manifest logic, pipe-axis projection |
| Tracking preparation | YOLO target selection, position/velocity smoothing, bounded local-candidate hold logic |
| Dataset tooling | Read-only YOLO dataset validation and fixed-shape training/export helpers |
| Engineering record | Measured FPS/accuracy comparison, deployment decision, calibration boundary and stop conditions |

## Measured results

The values below come from local experiment records. They are setup-specific measurements, not guarantees for every camera or lighting condition.

| Candidate | Offline validation | K230 local-LCD observation | Decision |
| --- | --- | --- | --- |
| Full frame, 352 × 192 | mAP50 0.977; mAP50-95 0.817 | about 28 FPS | Initial control candidate |
| Full frame, 512 × 288 | mAP50 0.983; mAP50-95 0.834 | about 20 FPS | Accuracy/debug reference |
| Centre ROI, 512 × 160 | mAP50 0.979; mAP50-95 0.787 | about 12 FPS; zero detections in tested K230 scene | Do not deploy this version |

The offline validation set is limited and the ROI geometry did not match the tested K230 camera view. This project records the failure boundary instead of presenting the ROI result as a finished optimization.

## Repository map

```text
src/
  control/                  K230 integration, position, velocity and UART preparation
tools/                      host-side fixed-shape and dataset utilities
docs/
  ARCHITECTURE.md            data flow and module boundaries
  VALIDATION.md              measurements and interpretation limits
  CONTROL_HANDOFF.md         calibration and control-integration boundary
```

## How to read it

1. Read [the architecture](docs/ARCHITECTURE.md) to understand the data path.
2. Read [the validation record](docs/VALIDATION.md) before quoting metrics.
3. Read `src/control/ball_axis.py` and `hybrid_selector.py` for portable logic that does not depend on K230 APIs.
4. Review `src/control/01_yolo352_position_uart.py`. It defaults to `UART_ENABLED = False` and `CALIBRATION_READY = False`; it emits diagnostic lines and does not drive a motor.
5. Treat any higher-rate camera experiment as a separate board-side experiment. It must not replace the stable pipeline unless cadence and coordinate calibration are rechecked.

## Hardware-side dependencies

The board scripts expect the K230/CanMV runtime, `libs.PipeLine`, `libs.YOLO`, `libs.Utils`, `media.*`, and a model copied separately to the board. Those vendor/runtime files and model artefacts are intentionally not committed here.

The diagnostic line prepared for a future MCU link is:

```text
BALL,seq=...,valid=...,source=yolo,cx_px=...,cy_px=...,pos_mm=...,vx_mmps=...
```

The actual K230 pins and MSPM0 wire pair must be confirmed on hardware before enabling UART. The current source keeps that path disabled by default.

## Scope and licensing

This is a curated public release. See [NOTICE.md](NOTICE.md) for the boundary around datasets, weights, vendor components and the unfinished closed loop.

