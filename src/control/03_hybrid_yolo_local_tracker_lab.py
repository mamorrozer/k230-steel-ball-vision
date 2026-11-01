# -*- coding: utf-8 -*-
"""Single-pipeline hybrid framework. Default is the proven YOLO-only baseline.

The local RGB565 extractor is intentionally disabled until it is verified on the
current v1.8 firmware. This prevents the old multi-channel deadlock from being
mistaken for a usable control path.
"""

from libs.PipeLine import PipeLine
from libs.YOLO import YOLOv8
from hybrid_selector import HybridTracker, select_best_yolo
import time

rgb888p_size = [352, 192]
kmodel_path = "/sdcard/kmodel/steel_ball_pipe_task_combined_v1_20260730_normal_352x192.kmodel"
labels = ["steel_ball"]
conf_thresh = 0.25
nms_thresh = 0.45
max_boxes_num = 10

MODE = "YOLO_ONLY"  # YOLO_ONLY is verified path. HYBRID_EXPERIMENT requires local adapter.
YOLO_EVERY_N = 4
LOCAL_ADAPTER_READY = False


def local_candidate_from_current_frame(_frame, _tracker):
    # Add the tested RGB565/find_blobs adapter here only after it has passed the
    # dedicated single-channel test in 04. Returning None is intentional today.
    return None


if __name__ == "__main__":
    pl = PipeLine(rgb888p_size=rgb888p_size, display_mode="lcd", debug_mode=0)
    pl.create()
    yolo = YOLOv8(task_type="detect", mode="video", kmodel_path=kmodel_path,
                  labels=labels, rgb888p_size=rgb888p_size, model_input_size=rgb888p_size,
                  conf_thresh=conf_thresh, nms_thresh=nms_thresh, max_boxes_num=max_boxes_num,
                  debug_mode=0)
    tracker = HybridTracker()
    sequence = 0
    local_unavailable = 0
    last_log = time.ticks_ms()
    print("HYBRID,start,mode=%s,local_adapter_ready=%d" % (MODE, LOCAL_ADAPTER_READY))
    try:
        while True:
            now = time.ticks_ms()
            frame = pl.get_frame()
            run_yolo = MODE == "YOLO_ONLY" or sequence % YOLO_EVERY_N == 0
            if run_yolo:
                detections = yolo.run(frame)
                yolo.draw_result(pl, detections)
                tracker.update_yolo(select_best_yolo(detections, conf_thresh), now)
            elif MODE == "HYBRID_EXPERIMENT" and LOCAL_ADAPTER_READY:
                tracker.update_local(local_candidate_from_current_frame(frame, tracker), now)
            elif MODE == "HYBRID_EXPERIMENT":
                local_unavailable += 1
            result, source = tracker.result(now)
            if result is not None:
                pl.osd_img.draw_string_advanced(6, 6, 18, "src:%s q:%.2f" %
                                                 (source, result.get("quality", 0.0)),
                                                 color=(255, 255, 255, 0))
            if time.ticks_diff(now, last_log) >= 1000:
                print("HYBRID,mode=%s,source=%s,valid=%d,local_unavailable=%d" %
                      (MODE, source, result is not None, local_unavailable))
                local_unavailable = 0
                last_log = now
            pl.show_image()
            sequence += 1
    finally:
        yolo.deinit()
        pl.destroy()
