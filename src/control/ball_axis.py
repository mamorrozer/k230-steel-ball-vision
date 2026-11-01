"""Pipe-axis coordinate and smoothing helpers. Pure Python/MicroPython compatible."""


def _sqrt(value):
    return value ** 0.5


class PipeAxisCalibration:
    def __init__(self, left_px, left_mm, right_px, right_mm, max_axis_error_px):
        self.left_px = left_px
        self.left_mm = left_mm
        self.right_px = right_px
        self.right_mm = right_mm
        self.max_axis_error_px = max_axis_error_px
        dx = right_px[0] - left_px[0]
        dy = right_px[1] - left_px[1]
        self._length2 = dx * dx + dy * dy
        self._dx = dx
        self._dy = dy

    def ready(self):
        return self._length2 > 1 and self.max_axis_error_px > 0

    def measure(self, cx, cy):
        if not self.ready():
            return {"valid": 0, "reason": "cal_not_ready"}

        rx = cx - self.left_px[0]
        ry = cy - self.left_px[1]
        t = (rx * self._dx + ry * self._dy) / self._length2
        foot_x = self.left_px[0] + t * self._dx
        foot_y = self.left_px[1] + t * self._dy
        err = _sqrt((cx - foot_x) ** 2 + (cy - foot_y) ** 2)
        position_mm = self.left_mm + t * (self.right_mm - self.left_mm)
        valid = int(err <= self.max_axis_error_px and 0.0 <= t <= 1.0)
        return {
            "valid": valid,
            "reason": "ok" if valid else "off_axis_or_outside",
            "position_mm": position_mm,
            "axis_error_px": err,
            "t": t,
            "foot_x": foot_x,
            "foot_y": foot_y,
        }


class PositionVelocityFilter:
    def __init__(self, alpha=0.45, velocity_alpha=0.30):
        self.alpha = alpha
        self.velocity_alpha = velocity_alpha
        self.position_mm = None
        self.velocity_mmps = 0.0
        self.last_ms = None

    def reset(self):
        self.position_mm = None
        self.velocity_mmps = 0.0
        self.last_ms = None

    def update(self, position_mm, now_ms):
        if self.position_mm is None or self.last_ms is None:
            self.position_mm = position_mm
            self.last_ms = now_ms
            return self.position_mm, self.velocity_mmps

        dt_ms = now_ms - self.last_ms
        if dt_ms <= 0:
            return self.position_mm, self.velocity_mmps
        raw_velocity = (position_mm - self.position_mm) * 1000.0 / dt_ms
        self.position_mm += self.alpha * (position_mm - self.position_mm)
        self.velocity_mmps += self.velocity_alpha * (raw_velocity - self.velocity_mmps)
        self.last_ms = now_ms
        return self.position_mm, self.velocity_mmps
