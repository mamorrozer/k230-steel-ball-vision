"""State-only YOLO/local tracker. Camera-specific local extraction stays outside."""


def center_of(box):
    return ((box[2] + box[0]) * 0.5, (box[3] + box[1]) * 0.5)


def select_best_yolo(detections, min_conf):
    best = None
    for det in detections:
        if len(det) < 6 or det[1] < min_conf:
            continue
        if best is None or det[1] > best["quality"]:
            best = {
                "class_id": det[0], "quality": det[1],
                "x1": det[2], "y1": det[3], "x2": det[4], "y2": det[5],
                "cx": (det[2] + det[4]) * 0.5,
                "cy": (det[3] + det[5]) * 0.5,
            }
    return best


class HybridTracker:
    """YOLO is authoritative; local candidates are accepted only near its last box."""
    def __init__(self, hold_ms=180, local_radius_px=45):
        self.hold_ms = hold_ms
        self.local_radius_px = local_radius_px
        self.last = None
        self.last_yolo_ms = None
        self.source = "none"

    def update_yolo(self, detection, now_ms):
        if detection is None:
            return None
        self.last = detection.copy()
        self.last_yolo_ms = now_ms
        self.last["timestamp_ms"] = now_ms
        self.source = "yolo"
        return self.last

    def update_local(self, candidate, now_ms):
        if candidate is None or self.last is None:
            return None
        dx = candidate["cx"] - self.last["cx"]
        dy = candidate["cy"] - self.last["cy"]
        if dx * dx + dy * dy > self.local_radius_px * self.local_radius_px:
            return None
        accepted = candidate.copy()
        accepted["timestamp_ms"] = now_ms
        accepted["quality"] = candidate.get("quality", 0.0)
        self.last = accepted
        self.source = "local"
        return accepted

    def result(self, now_ms):
        if self.last is None:
            return None, "lost"
        age_ms = now_ms - self.last.get("timestamp_ms", now_ms)
        if age_ms > self.hold_ms:
            self.last = None
            self.source = "none"
            return None, "lost"
        return self.last, self.source if age_ms == 0 else "hold"
