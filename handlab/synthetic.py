"""Deterministic, explicitly synthetic hand poses, not real MediaPipe accuracy data."""
import numpy as np


def hand():
    xy = [[.5,.65], [.44,.61],[.4,.56],[.37,.51],[.34,.47],
          [.44,.48],[.43,.39],[.43,.32],[.43,.26],
          [.50,.46],[.50,.36],[.50,.28],[.50,.21],
          [.56,.48],[.57,.39],[.58,.32],[.59,.27],
          [.62,.51],[.65,.44],[.67,.39],[.69,.35]]
    return np.column_stack([xy, np.zeros(21)])


def sequence(scenario='stationary', frames=240, fps=30, seed=7):
    rng = np.random.default_rng(seed)
    base = hand()
    for i in range(frames):
        t = i / fps
        truth = base.copy()
        if scenario in ('slow', 'fast', 'whole_jump', 'occlusion', 'partial'):
            frequency = 1.5 if scenario == 'fast' else .25
            truth[:, 0] += .12 * np.sin(2*np.pi*frequency*t)
        if scenario == 'fingers':
            for indices in ([6,7,8],[10,11,12],[14,15,16],[18,19,20]):
                anchor = truth[indices[0]-1].copy()
                angle = .6 * np.sin(2*np.pi*1.5*t)
                r = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                truth[indices,:2] = (truth[indices,:2]-anchor[:2]) @ r.T + anchor[:2]
        observed = truth + rng.normal(0,.0015,(21,3))
        if scenario == 'joint_jump' and i == 60:
            observed[8] += [.4,-.4,.3]
        if scenario == 'whole_jump' and i == 60:
            observed += [.5,0,0]
        if scenario == 'partial' and 60 <= i < 66:
            observed[8] += [.35,-.3,.2]
        if scenario == 'occlusion':
            if 60 <= i < 72:
                observed = None
            elif i >= 72:
                truth += [.12,-.03,.02]
                observed = truth + rng.normal(0,.0015,(21,3))
        yield t, observed, truth
