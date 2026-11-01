# -*- coding: utf-8 -*-
"""K230: YOLO position baseline. Display and UART are diagnostics only; no motors."""

from libs.PipeLine import PipeLine
from libs.YOLO import YOLOv8
from libs.Utils import ScopedTiming
from ball_axis import PipeAxisCalibration, PositionVelocityFilter
from hybrid_selector import select_best_yolo

import time
import gc

rgb888p_size = [352, 192]
display_mode = "lcd"
kmodel_path = "/sdcard/kmodel/steel_ball_pipe_task_combined_v1_20260730_normal_352x192.kmodel"
labels = ["steel_ball"]
conf_thresh = 0.25
nms_thresh = 0.45
max_boxes_num = 10
debug_mode = 0

# Replace all five values after drawing two known position marks on the final pipe.
CALIBRATION_READY = False
LEFT_MARK_PX = (0, 0)
LEFT_MARK_MM = -125.0
RIGHT_MARK_PX = (0, 0)
RIGHT_MARK_MM = 125.0
MAX_AXIS_ERROR_PX = 18.0

# Hardware UART remains off until its actual K230 pins and the MSPM0 wire pair are verified.
UART_ENABLED = False
UART_ID = 1
UART_TX_PIN = 9
UART_RX_PIN = 10
UART_BAUD = 115200
UART_PERIOD_MS = 40
LOG_PERIOD_MS = 500


def open_uart():
    if not UART_ENABLED:
        return None
    from machine import UART, FPIOA
    fpioa = FPIOA()
    fpioa.set_function(UART_TX_PIN, fpioa.UART1_TXD, ie=0, oe=1, pu=1)
    fpioa.set_function(UART_RX_PIN, fpioa.UART1_RXD, ie=1, oe=0, pu=1)
    return UART(UART_ID, UART_BAUD)


def draw_cross(osd, x, y, color):
    osd.draw_line(int(x) - 5, int(y), int(x) + 5, int(y), color=color, thickness=2)
    osd.draw_line(int(x), int(y) - 5, int(x), int(y) + 5, color=color, thickness=2)


if __name__ == "__main__":
    calibration = PipeAxisCalibration(LEFT_MARK_PX, LEFT_MARK_MM, RIGHT_MARK_PX,
                                      RIGHT_MARK_MM, MAX_AXIS_ERROR_PX)
    filter_state = PositionVelocityFilter()
    uart = open_uart()
    pl = PipeLine(rgb888p_size=rgb888p_size, display_mode=display_mode, debug_mode=debug_mode)
    pl.create()
    yolo = YOLOv8(task_type="detect", mode="video", kmodel_path=kmodel_path,
                  labels=labels, rgb888p_size=rgb888p_size,
                  model_input_size=rgb888p_size, conf_thresh=conf_thresh,
                  nms_thresh=nms_thresh, max_boxes_num=max_boxes_num,
                  debug_mode=debug_mode)
    last_log_ms = 0
    last_uart_ms = 0
    sequence = 0
    print("BALL_BASELINE,model=352x192,uart_enabled=%d,cal_ready=%d" %
          (UART_ENABLED, CALIBRATION_READY))
    try:
        while True:
            now_ms = time.ticks_ms()
            with ScopedTiming("yolo", debug_mode > 0):
                frame = pl.get_frame()
                detections = yolo.run(frame)
                yolo.draw_result(pl, detections)

            target = select_best_yolo(detections, conf_thresh)
            measurement = {"valid": 0, "reason": "no_ball"}
            pos_mm = 0.0
            velocity_mmps = 0.0
            if target is not None:
                draw_cross(pl.osd_img, target["cx"], target["cy"], color=(255, 255, 0, 0))
                if CALIBRATION_READY:
                    measurement = calibration.measure(target["cx"], target["cy"])
                    if measurement["valid"]:
                        pos_mm, velocity_mmps = filter_state.update(measurement["position_mm"], now_ms)
                else:
                    measurement = {"valid": 0, "reason": "cal_not_ready", "axis_error_px": -1.0}

            status = "BALL,seq=%d,valid=%d,source=yolo,cx_px=%d,cy_px=%d,pos_mm=%.1f,vx_mmps=%.1f,axis_err_px=%.1f,quality=%.3f,reason=%s" % (
                sequence, measurement["valid"], int(target["cx"]) if target else -1,
                int(target["cy"]) if target else -1, pos_mm, velocity_mmps,
                measurement.get("axis_error_px", -1.0), target["quality"] if target else 0.0,
                measurement["reason"])
            if now_ms - last_log_ms >= LOG_PERIOD_MS:
                print(status)
                last_log_ms = now_ms
            if uart is not None and now_ms - last_uart_ms >= UART_PERIOD_MS:
                try:
                    uart.write(status + "\n")
                except Exception as error:
                    print("BALL_UART,error=%s" % error)
                last_uart_ms = now_ms
            pl.show_image()
            sequence += 1
            if sequence % 120 == 0:
                gc.collect()
    finally:
        yolo.deinit()
        pl.destroy()
