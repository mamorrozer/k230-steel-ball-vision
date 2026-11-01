# Control handoff boundary

The vision side prepares an observation for a future MCU controller. It does not directly drive a motor in this repository.

## Calibration

Set two known marks on the final pipe and record their image coordinates and physical positions. Fill the values in `src/control/01_yolo352_position_uart.py`, then verify left endpoint, midpoint, right endpoint and an off-axis point. A measurement is valid only when its projection remains inside the segment and its perpendicular pixel error is below the configured limit.

## Diagnostic output

The prepared line is ASCII and human-readable:

```text
BALL,seq=12,valid=1,source=yolo,cx_px=176,cy_px=94,pos_mm=3.2,vx_mmps=-8.4,axis_err_px=1.5,quality=0.91,reason=ok
```

`UART_ENABLED` is false by default. Confirm K230 FPIOA pins, baud rate, common ground and the MSPM0 UART input level before enabling it.

## What the MCU must still do

The receiving controller must reject malformed or stale lines, apply a timeout, bound actuator output, and enter a safe state when the vision link or target is invalid. Those behaviours belong to control firmware and are outside this vision-only release.

