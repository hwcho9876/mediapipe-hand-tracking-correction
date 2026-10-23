"""One hand stateful correction pipeline. No image inference or GUI dependencies."""
from dataclasses import dataclass, field
import logging
import time
import numpy as np
from .config import Config
from .geometry import valid_points, palm_scale, max_distance
from .filters import OneEuro
from .anomaly import AnomalyDetector, Assessment
from .correction import Corrector
from .state import State, StateMachine
from .recovery import Recovery

logger=logging.getLogger(__name__)


@dataclass
class Output:
    points: np.ndarray | None
    state: str
    reason: str
    event: str | None = None
    corrected: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    predicted: bool = False
    resynced: bool = False
    processing_ms: float = 0.


class Pipeline:
    def __init__(self, config=None):
        self.config=(config or Config()).validate()
        self.oef=OneEuro(self.config.min_cutoff,self.config.beta,self.config.d_cutoff,self.config.long_gap)
        self.detector=AnomalyDetector(self.config)
        self.corrector=Corrector(self.config)
        self.recovery=Recovery(self.config)
        self.machine=StateMachine(self.config.suspect_confirm)
        self.last_t=None
        self.last_valid_t=None
        self.last_output=None
        self.last_trusted_output=None
        self.missing=False
        self.blocked=False
        self.previous_oef_enabled=self.config.oef_enabled

    def set_config(self, config):
        self.config=config.validate()
        self.detector.config=self.corrector.config=self.recovery.config=self.config
        self.machine.confirmation=self.config.suspect_confirm

    def step(self, observed, t, aspect=1., handedness=None, handedness_score=None):
        started=time.perf_counter()
        def result(points,reason,event=None,assessment=None,predicted=False,resynced=False,extra=None):
            if points is not None:
                points=points.copy()
            self.last_output=points
            if event:
                logger.info('t=%.4f %s reason=%s',t,event,reason)
            return Output(points,self.machine.state.value,reason,event,
                np.flatnonzero(assessment.bad).tolist() if assessment and c.correction_enabled and c.detection_enabled else [],
                          {**(assessment.metrics if assessment else {}),**(extra or {})},predicted,resynced,
                          (time.perf_counter()-started)*1000)
        # Duplicate/reversed/nonfinite timestamps do not advance ANY tracking state.
        if not np.isfinite(t) or t<0 or (self.last_t is not None and t<=self.last_t):
            return Output(None if self.last_output is None else self.last_output.copy(),
                          self.machine.state.value,'invalid_timestamp',processing_ms=(time.perf_counter()-started)*1000)
        c=self.config
        elapsed=0 if self.last_t is None else t-self.last_t
        self.last_t=t
        if observed is not None:
            observed=np.asarray(observed,dtype=float)
        if not valid_points(observed):
            self.missing=True
            self.recovery.active=False
            event=self.machine.transition(State.LOST)
            pred=self.corrector.predict(t) if c.correction_enabled else None
            return result(pred,'not_detected' if observed is None else 'invalid_landmarks',event,predicted=pred is not None)
        # On reappearance, old kinematics cannot disprove a real changed hand position.
        if self.missing or elapsed>c.long_gap:
            self.detector.reset()
        assessment=self.detector.assess(observed,t,aspect,handedness,handedness_score) if c.detection_enabled else Assessment(reason='detection_disabled')
        suspicious=assessment.kind!='good'
        apply_gate=c.detection_enabled and c.correction_enabled
        if apply_gate and assessment.kind in ('uncertain','reject'):
            self.blocked=True
            self.recovery.active=False
            pred=self.corrector.predict(t)
            event=self.machine.suspect() if pred is not None else self.machine.transition(State.LOST)
            return result(pred,assessment.reason,event,assessment,predicted=pred is not None)
        valid=~assessment.bad if apply_gate else np.ones(21,bool)
        self.oef.min_cutoff,self.oef.beta,self.oef.d_cutoff,self.oef.max_gap=c.min_cutoff,c.beta,c.d_cutoff,c.long_gap
        filtered=self.oef.update(observed,t,valid)
        target=filtered if c.oef_enabled else observed.copy()
        # Maintain per-joint real-observation ages. Fabricated values never feed OEF or references.
        self.corrector.accept(observed,t,valid)
        if apply_gate and assessment.bad.any():
            repaired=self.corrector.repair(observed,assessment.bad,t)
            if repaired is None:
                self.blocked=True
                self.recovery.active=False
                event=self.machine.transition(State.LOST)
                return result(None,'local_prediction_expired',event,assessment)
            target[assessment.bad]=repaired[assessment.bad]
        self.detector.accept(observed,t,valid,handedness)
        scale=palm_scale(observed)
        gap=0 if self.last_valid_t is None else t-self.last_valid_t
        toggle=c.oef_enabled!=self.previous_oef_enabled
        need_recovery=self.missing or self.blocked or toggle or elapsed>c.long_gap
        self.previous_oef_enabled=c.oef_enabled
        self.last_valid_t=t
        self.missing=self.blocked=False
        resynced=False
        event=None
        anchor=self.last_trusted_output
        if need_recovery and anchor is not None:
            if gap>c.long_gap or max_distance(anchor,target)/scale>c.resync_distance:
                # Explicit discontinuity/reinitialization instead of pretending a long blind path is valid.
                self.recovery.active=False
                resynced=True
            elif c.ema_enabled:
                self.recovery.start(anchor,target,t)
                event=self.machine.transition(State.RECOVERING)
        if not c.ema_enabled:
            self.recovery.active=False
        extra={}
        if self.recovery.active:
            output,done,extra=self.recovery.step(target,t,scale)
            if done:
                event=self.machine.transition(State.NORMAL)
            else:
                event=event or self.machine.transition(State.RECOVERING)
        else:
            output=target
            event=self.machine.suspect() if suspicious else self.machine.normal()
        if not suspicious:
            self.last_trusted_output=output.copy()
        return result(output,'explicit_resync' if resynced else assessment.reason,event,assessment,
                      resynced=resynced,extra=extra)


def comparison_configs(base=None):
    base=(base or Config()).to_dict()
    return {name:Config.from_dict({**base,**flags}) for name,flags in {
        'raw':dict(oef_enabled=False,detection_enabled=False,correction_enabled=False,ema_enabled=False),
        'oef':dict(oef_enabled=True,detection_enabled=False,correction_enabled=False,ema_enabled=False),
        'corrected':dict(oef_enabled=True,detection_enabled=True,correction_enabled=True,ema_enabled=False),
        'full':dict(oef_enabled=True,detection_enabled=True,correction_enabled=True,ema_enabled=True),
    }.items()}
