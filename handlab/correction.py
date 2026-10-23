"""Short bounded extrapolation. Never used as a new trusted measurement."""
import numpy as np
from .geometry import palm_scale


class Corrector:
    def __init__(self, config):
        self.config = config
        self.points = None
        self.times = np.full(21,np.nan)
        self.velocity = np.zeros((21,3))

    def accept(self, points, t, valid=None):
        mask = np.ones(21,bool) if valid is None else valid
        if self.points is None:
            self.points = points.copy()
            self.times[mask] = t
            return
        dt = t-self.times
        update = mask & np.isfinite(dt) & (dt > 0)
        self.velocity[update] = (points[update]-self.points[update])/dt[update,None]
        # Do not extrapolate beyond plausible short displacement even after a rebaseline.
        limit = 4*palm_scale(points)
        norm = np.linalg.norm(self.velocity,axis=1,keepdims=True)
        self.velocity *= np.minimum(1,limit/np.maximum(norm,1e-9))
        self.points[mask] = points[mask]
        self.times[mask] = t

    def predict(self, t):
        if self.points is None or np.any(t-self.times > self.config.hold_seconds):
            return None
        dt = np.clip(t-self.times,0,self.config.prediction_seconds)
        delta = self.velocity*dt[:,None]
        limit = self.config.prediction_limit*palm_scale(self.points)
        norm = np.linalg.norm(delta,axis=1,keepdims=True)
        delta *= np.minimum(1,limit/np.maximum(norm,1e-9))
        return self.points+delta

    def repair(self, observed, bad, t):
        predicted = self.predict(t)
        if predicted is None:
            return None
        repaired = observed.copy()
        if (~bad).sum() >= 10:
            shift = np.median(observed[~bad]-self.points[~bad],axis=0)
            limit = self.config.prediction_limit*palm_scale(self.points)
            shift *= min(1,limit/max(np.linalg.norm(shift),1e-9))
            predicted = self.points+shift
        repaired[bad] = predicted[bad]
        return repaired
