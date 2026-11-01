# -*- coding: utf-8 -*-
"""Standalone single-channel LAB/find_blobs diagnostic for K230.

Green boxes: all blobs which pass shape filters.
Yellow box: one continuity-tracked candidate.
This is a measurement tool, not an asserted steel-ball detector.
"""

from media.sensor import Sensor
from media.display import Display
from media.media import MediaManager
from cv_blob_logic import PROFILES, evaluate, BlobContinuityTracker

import time
import gc

PROFILE_NAME = "white_l33_70"
ROI = [0, 0, 640, 480]  # Replace with final pipe rectangle: [x, y, width, height].
ROI_READY = False
CAMERA_SIZE = (640, 480)
DISPLAY_SIZE = (640, 480)
LOG_PERIOD_MS = 1000


def _call_or_none(blob, name):
    try:
        value = getattr(blob, name)
        return value() if callable(value) else value
    except Exception:
        return None


def blob_feature(blob):
    x = blob[0]
    y = blob[1]
    w = blob[2]
    h = blob[3]
    pixels = _call_or_none(blob, "pixels")
    fill = _call_or_none(blob, "density")
    if fill is None and pixels is not None:
        fill = pixels / max(1, w * h)
    if fill is None:
        fill = 0.0
    perimeter = _call_or_none(blob, "perimeter")
    circle = -1.0
    if perimeter is not None and perimeter > 0:
        circle = 12.5663706 * w * h / (perimeter * perimeter)
    return {"x": x, "y": y, "w": w, "h": h, "cx": x + w * 0.5,
            "cy": y + h * 0.5, "area": w * h, "fill": fill, "circle": circle}


if __name__ == "__main__":
    profile = PROFILES[PROFILE_NAME]
    sensor = Sensor(id=2, width=1280, height=960, fps=90)
    sensor.reset()
    sensor.set_framesize(width=CAMERA_SIZE[0], height=CAMERA_SIZE[1])
    sensor.set_pixformat(Sensor.RGB565)
    Display.init(Display.ST7701, width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1], to_ide=True)
    MediaManager.init()
    tracker = BlobContinuityTracker()
    last_log = time.ticks_ms()
    counts = {"raw": 0, "area": 0, "aspect": 0, "fill": 0, "circle": 0, "pass": 0, "frames": 0}
    print("CVLAB,start,profile=%s,profile_ready=%d,roi_ready=%d,lab=%s" %
          (PROFILE_NAME, profile["ready"], ROI_READY, profile["lab"]))
    try:
        while True:
            now = time.ticks_ms()
            img = sensor.snapshot()
            blobs = img.find_blobs([profile["lab"]], roi=ROI, area_threshold=1, merge=False)
            candidates = []
            for blob in blobs:
                counts["raw"] += 1
                feature = blob_feature(blob)
                verdict = evaluate(feature, profile)
                if verdict["area_ok"]:
                    counts["area"] += 1
                if verdict["area_ok"] and verdict["aspect_ok"]:
                    counts["aspect"] += 1
                if verdict["area_ok"] and verdict["aspect_ok"] and verdict["fill_ok"]:
                    counts["fill"] += 1
                if verdict["pass"]:
                    counts["circle"] += 1
                    counts["pass"] += 1
                    feature["quality"] = min(1.0, feature["fill"])
                    candidates.append(feature)
                    img.draw_rectangle(feature["x"], feature["y"], feature["w"], feature["h"],
                                       color=(0, 255, 0), thickness=2)
            tracked, state = tracker.update(candidates, now)
            if tracked is not None:
                img.draw_rectangle(int(tracked["x"]), int(tracked["y"]), int(tracked["w"]), int(tracked["h"]),
                                   color=(255, 255, 0), thickness=2)
                img.draw_string_advanced(4, 4, 18, "track:%s" % state, color=(255, 255, 0))
            else:
                img.draw_string_advanced(4, 4, 18, "track:lost", color=(255, 0, 0))
            img.draw_rectangle(ROI[0], ROI[1], ROI[2], ROI[3], color=(0, 0, 255), thickness=1)
            Display.show_image(img)
            counts["frames"] += 1
            elapsed = time.ticks_diff(now, last_log)
            if elapsed >= LOG_PERIOD_MS:
                fps = counts["frames"] * 1000.0 / elapsed
                print("CVLAB,fps=%.1f,raw=%d,area=%d,aspect=%d,fill=%d,circle=%d,pass=%d,track=%s" %
                      (fps, counts["raw"], counts["area"], counts["aspect"], counts["fill"],
                       counts["circle"], counts["pass"], state))
                counts = {"raw": 0, "area": 0, "aspect": 0, "fill": 0, "circle": 0, "pass": 0, "frames": 0}
                last_log = now
                gc.collect()
    finally:
        sensor.stop()
        Display.deinit()
        MediaManager.deinit()
