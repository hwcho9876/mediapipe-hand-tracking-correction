"""Shared-alpha pose EMA and a bounded, output-only handoff to independent OEF."""
import numpy as np
from .geometry import max_distance


class Recovery:
    def __init__(self, config):
        self.config = config
        self.active = False
        self.value = None
        self.previous_target = None
        self.confirmed = 0
        self.bridge_start = None
        self.bridge_offset = None
        self.bridge_duration = .15

    def start(self, anchor, target, t):
        self.active = True
        self.value = anchor.copy()
        self.previous_target = target.copy()
        self.start_time = self.last_t = t
        self.confirmed = 0
        self.bridge_start = None

    def step(self, target, t, scale):
        dt = t-self.last_t
        if dt <= 0:
            return self.value.copy(), False, {'recovery_mode':'anchor'}
        c=self.config
        old=self.value.copy()
        if self.bridge_start is not None:
            u=min(1.,(t-self.bridge_start)/self.bridge_duration)
            weight=1-(3*u*u-2*u*u*u)
            self.value=target+weight*self.bridge_offset
            done=u>=1
            mode='timeout_bridge'
        else:
            distance=max_distance(self.value,target)/scale
            # The same coefficient updates root translation AND all relative joint offsets.
            # Large errors accelerate recovery, but never modify OEF's internal state.
            tau=c.ema_tau/max(1.,min(4.,distance))
            a=-np.expm1(-dt/tau)
            self.value += a*(target-self.value)
            gap=max_distance(self.value,target)/scale
            mismatch=max_distance((self.value-old)/dt,(target-self.previous_target)/dt)/scale
            self.confirmed=self.confirmed+1 if gap<c.recovery_distance and mismatch<c.recovery_velocity else 0
            done=self.confirmed>=c.recovery_confirm
            mode='ema'
            if not done and t-self.start_time >= c.recovery_timeout:
                self.bridge_start=t
                self.bridge_offset=self.value-target
                mode='timeout_bridge_start'
        residual=max_distance(self.value,target)/scale
        if done:
            self.value=target.copy()
            self.active=False
        self.last_t=t
        self.previous_target=target.copy()
        return self.value.copy(),done,{'recovery_mode':mode,'recovery_gap':residual,
                                      'recovery_elapsed':t-self.start_time}
