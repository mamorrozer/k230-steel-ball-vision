# -*- coding: utf-8 -*-
"""K230 pipeline cadence experiment. It measures loop cadence, not object detection."""

from libs.PipeLine import PipeLine
import time
import gc

PROFILE_NAME = "352x192"
PROFILES = {
    "352x192": [352, 192],
    "288x160": [288, 160],
    "512x288": [512, 288],
}
SHOW_IMAGE = True
LOG_PERIOD_MS = 1000


if __name__ == "__main__":
    size = PROFILES[PROFILE_NAME]
    pl = PipeLine(rgb888p_size=size, display_mode="lcd", debug_mode=0)
    pl.create()
    count = 0
    last_log = time.ticks_ms()
    last_frame = last_log
    worst_ms = 0
    print("CADENCE,start,profile=%s,size=%dx%d,show=%d" %
          (PROFILE_NAME, size[0], size[1], SHOW_IMAGE))
    try:
        while True:
            now = time.ticks_ms()
            frame_ms = time.ticks_diff(now, last_frame)
            last_frame = now
            if frame_ms > worst_ms:
                worst_ms = frame_ms
            pl.get_frame()
            if SHOW_IMAGE:
                pl.osd_img.draw_string_advanced(6, 6, 18, "cadence", color=(255, 255, 0, 0))
                pl.show_image()
            count += 1
            elapsed = time.ticks_diff(now, last_log)
            if elapsed >= LOG_PERIOD_MS:
                fps = count * 1000.0 / elapsed
                print("CADENCE,fps=%.1f,frames=%d,worst_frame_ms=%d,show=%d" %
                      (fps, count, worst_ms, SHOW_IMAGE))
                count = 0
                worst_ms = 0
                last_log = now
                gc.collect()
    finally:
        pl.destroy()
