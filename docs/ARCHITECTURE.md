# Architecture

## Runtime path

```text
K230 camera
    ├─ display channel ───────────────► local LCD / optional IDE mirror
    └─ RGB888 planar channel
          ▼
      YOLO model (352×192 or 512×288)
          ▼ detections
      select_best_yolo()
          ▼ best target box and confidence
      PipeAxisCalibration.measure()
          ▼ position_mm + axis error
      PositionVelocityFilter.update()
          ▼ smoothed position / velocity
      diagnostic overlay + BALL UART line
```

## Module boundaries

- `src/control/01_yolo352_position_uart.py` is the K230 integration example. It calls the board runtime but keeps UART and calibration disabled by default.
- `src/control/ball_axis.py` contains pure geometry and smoothing logic without camera or motor APIs.
- `src/control/hybrid_selector.py` contains state-only target selection. YOLO remains authoritative; a local candidate can only be accepted near the last YOLO box and within a short hold window.
- `src/control/cv_blob_logic.py` is diagnostic filtering code. Its profiles are marked `ready=0` because pure-CV detection was not reliable enough for control.
- `src/control/01_yolo352_position_uart.py` logs results and prepares diagnostic UART output, but does not command motors.

## Coordinate contract

Host tooling uses `(height, width)` ordering while K230 model/camera APIs use `(width, height)`. The conversion is explicit to avoid swapping 512×288 and 288×512. Position is invalid until two physical marks provide a pixel-to-mm calibration and an acceptable axis error.

## Safety boundary

The vision result is an observation. It becomes a control input only after final-pipe camera calibration, measurements of misses/false detections/jitter, UART pin and electrical verification, parser timeout and invalid-frame handling, and an independent motor stop path.

