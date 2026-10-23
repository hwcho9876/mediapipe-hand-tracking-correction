"""Heuristic evidence, not a ground-truth occlusion classifier."""
from dataclasses import dataclass, field
import numpy as np
from .geometry import PALM, EDGE_A, EDGE_B, center, palm_scale


@dataclass
class Assessment:
    kind: str = 'good'  # good / local / uncertain / reject
    bad: np.ndarray = field(default_factory=lambda: np.zeros(21, dtype=bool))
    metrics: dict = field(default_factory=dict)
    reason: str = 'accepted'


class AnomalyDetector:
    def __init__(self, config):
        self.config = config
        self.previous = None
        self.previous_t = None
        self.velocity = np.zeros(3)
        self.candidate = None
        self.candidate_count = 0
        self.handedness = None

    def reset(self):
        self.__init__(self.config)

    def assess(self, points, t, aspect=1., handedness=None, handedness_score=None):
        if self.previous is None:
            return Assessment(reason='first_observation')
        c = self.config
        dt = t - self.previous_t
        if dt <= 0 or dt > c.long_gap:
            return Assessment(reason='new_observation_epoch')
        old = self.previous
        scale = palm_scale(old)
        shifts = points - old
        translation = np.median(shifts, axis=0)
        residual = np.linalg.norm(shifts-translation, axis=1) / scale
        old_lengths = np.maximum(np.linalg.norm(old[EDGE_A]-old[EDGE_B],axis=1), .01*scale)
        new_lengths = np.linalg.norm(points[EDGE_A]-points[EDGE_B],axis=1)
        # Compensate uniform hand scale changes; this does not reconstruct true bone lengths.
        ratios = new_lengths / old_lengths / np.clip(palm_scale(points)/scale, .5, 2.)
        changed = (ratios > c.bone_ratio) | (ratios < 1/c.bone_ratio)
        bad = np.zeros(21,bool)
        for a,b in zip(EDGE_A[changed], EDGE_B[changed]):
            idx = a if residual[a] > residual[b] else b
            if residual[idx] > c.joint_residual:
                bad[idx] = True
        palm_shift = center(points)-center(old)
        velocity = palm_shift/dt
        speed = float(np.linalg.norm(velocity)/scale)
        acceleration = float(np.linalg.norm(velocity-self.velocity)/dt/scale)
        innovation = float(np.linalg.norm(palm_shift-self.velocity*dt)/scale)
        wrist_step = float(np.linalg.norm(shifts[0])/scale)
        border = float(np.mean((points[:,0] < -c.screen_margin) | (points[:,0] > 1+c.screen_margin) |
                               (points[:,1] < -c.screen_margin) | (points[:,1] > aspect+c.screen_margin)))
        hand_flip = bool(handedness and self.handedness and handedness != self.handedness and
                         handedness_score is not None and handedness_score >= .8)
        metrics = dict(speed=speed, acceleration=acceleration, innovation=innovation,
                       wrist_step=wrist_step, structure_changed=float(changed.mean()),
                       border_fraction=border, handedness_flip=hand_flip,
                       max_joint_residual=float(residual.max()))
        # A sudden STOP after a confirmed fast move must not become a second jump.
        temporal = innovation > c.jump_distance and speed > c.jump_speed and acceleration > c.acceleration
        severe = bad.sum() >= 8 or (changed.mean() > .5 and temporal)
        if severe:
            return Assessment('reject', np.ones(21,bool), metrics, 'structure_and_temporal_conflict')
        if temporal or (hand_flip and innovation > c.jump_distance/2) or (border > .6 and innovation > .3):
            # A rigid jump is ambiguous. Buffer briefly; persistent coherent evidence wins.
            consistent = self.candidate is not None and np.linalg.norm(center(points)-center(self.candidate))/scale < c.jump_distance
            self.candidate_count = self.candidate_count + 1 if consistent else 1
            self.candidate = points.copy()
            if self.candidate_count < c.suspect_confirm:
                return Assessment('uncertain', np.ones(21,bool), metrics, 'coherent_jump_unconfirmed')
            if bad.any():
                return Assessment('local', bad, metrics, 'confirmed_motion_with_local_error')
            return Assessment('good', bad, metrics, 'coherent_motion_confirmed')
        self.candidate = None
        self.candidate_count = 0
        return Assessment('local' if bad.any() else 'good', bad, metrics,
                          'local_structure_conflict' if bad.any() else 'accepted')

    def accept(self, points, t, valid=None, handedness=None):
        valid = np.ones(21,bool) if valid is None else valid
        if self.previous is None:
            self.previous = points.copy()
            self.previous_t = t
        else:
            dt = t-self.previous_t
            # Do not compute palm velocity using fabricated palm landmarks.
            if dt > 0 and valid[PALM].all():
                self.velocity = (center(points)-center(self.previous))/dt
            self.previous[valid] = points[valid]
            self.previous_t = t
        if handedness:
            self.handedness = handedness
