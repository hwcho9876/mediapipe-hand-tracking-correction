"""Independent per-landmark One Euro state; no EMA state is accepted here."""
import numpy as np


def alpha(cutoff, dt):
    return 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff * dt))


class OneEuro:
    def __init__(self, min_cutoff=1.5, beta=.4, d_cutoff=1., max_gap=1.):
        self.min_cutoff, self.beta, self.d_cutoff, self.max_gap = min_cutoff, beta, d_cutoff, max_gap
        self.raw = np.zeros((21, 3))
        self.value = np.zeros((21, 3))
        self.derivative = np.zeros((21, 3))
        self.last_t = np.full(21, np.nan)
        self.last_call = None

    def update(self, points, t, valid=None):
        if not np.isfinite(t) or (self.last_call is not None and t <= self.last_call):
            return self.value.copy()
        points = np.asarray(points, dtype=float)
        if points.shape != (21, 3):
            raise ValueError('OneEuro 입력은 (21,3) 배열이어야 합니다.')
        if not all(np.isfinite([self.min_cutoff, self.beta, self.d_cutoff, self.max_gap])) or min(self.min_cutoff, self.d_cutoff, self.max_gap) <= 0 or self.beta < 0:
            raise ValueError('OEF cutoff/max_gap은 양수, beta는 0 이상이어야 합니다.')
        self.last_call = t
        mask = np.isfinite(points).all(axis=1)
        if valid is not None:
            mask &= np.asarray(valid, dtype=bool)
        elapsed = t - self.last_t
        reset = mask & (~np.isfinite(self.last_t) | (elapsed > self.max_gap))
        self.value[reset] = self.raw[reset] = points[reset]
        self.derivative[reset] = 0
        regular = mask & ~reset
        dt = elapsed[regular, None]
        if regular.any():
            velocity = (points[regular] - self.raw[regular]) / dt
            ad = alpha(self.d_cutoff, dt)
            d = ad * velocity + (1 - ad) * self.derivative[regular]
            cutoff = self.min_cutoff + self.beta * np.abs(d)
            a = alpha(cutoff, dt)
            self.value[regular] += a * (points[regular] - self.value[regular])
            self.derivative[regular] = d
            self.raw[regular] = points[regular]
        self.last_t[mask] = t
        return self.value.copy()
