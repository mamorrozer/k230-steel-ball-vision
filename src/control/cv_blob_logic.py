"""Feature filtering and continuity logic for the standalone LAB blob experiment."""


PROFILES = {
    # Only L=33..70 was measured on the current white-pipe setup. Other limits
    # are deliberately broad until a real sample is measured on the final rig.
    "white_l33_70": {
        "ready": 0,
        "lab": (33, 70, -128, 127, -128, 127),
        "area_min": 6, "area_max": 4000,
        "aspect_min": 0.45, "aspect_max": 2.20,
        "fill_min": 0.20, "circle_min": 0.0,
    },
    "green_pending_measurement": {
        "ready": 0,
        "lab": (0, 100, -128, 127, -128, 127),
        "area_min": 6, "area_max": 4000,
        "aspect_min": 0.45, "aspect_max": 2.20,
        "fill_min": 0.20, "circle_min": 0.0,
    },
}


def evaluate(feature, profile):
    w = max(1, feature["w"])
    h = max(1, feature["h"])
    area = feature.get("area", w * h)
    aspect = w / h
    fill = feature.get("fill", 0.0)
    circle = feature.get("circle", -1.0)
    area_ok = profile["area_min"] <= area <= profile["area_max"]
    aspect_ok = profile["aspect_min"] <= aspect <= profile["aspect_max"]
    fill_ok = fill >= profile["fill_min"]
    circle_ok = profile["circle_min"] <= 0.0 or circle < 0.0 or circle >= profile["circle_min"]
    return {
        "area": area, "aspect": aspect, "fill": fill, "circle": circle,
        "area_ok": area_ok, "aspect_ok": aspect_ok,
        "fill_ok": fill_ok, "circle_ok": circle_ok,
        "pass": area_ok and aspect_ok and fill_ok and circle_ok,
    }


class BlobContinuityTracker:
    def __init__(self, jump_px=40, hold_ms=150):
        self.jump_px = jump_px
        self.hold_ms = hold_ms
        self.last = None
        self.last_ms = None

    def update(self, candidates, now_ms):
        best = None
        best_cost = None
        for candidate in candidates:
            if self.last is None:
                cost = -candidate.get("quality", 0.0)
            else:
                dx = candidate["cx"] - self.last["cx"]
                dy = candidate["cy"] - self.last["cy"]
                distance2 = dx * dx + dy * dy
                if distance2 > self.jump_px * self.jump_px:
                    continue
                cost = distance2 - candidate.get("quality", 0.0) * 10.0
            if best is None or cost < best_cost:
                best = candidate
                best_cost = cost
        if best is not None:
            self.last = best
            self.last_ms = now_ms
            return best, "track"
        if self.last is not None and now_ms - self.last_ms <= self.hold_ms:
            return self.last, "hold"
        self.last = None
        self.last_ms = None
        return None, "lost"
