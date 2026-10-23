"""All comparison arms consume exactly the same detector observations."""
from dataclasses import asdict
import time
from .config import Config
from .pipeline import Pipeline, comparison_configs


class Experiment:
    def __init__(self, config=None):
        self.segment=0
        self.frame=0
        self.configure(config or Config(),reset_segment=False)

    def configure(self, config,reset_segment=False):
        self.config=config.validate()
        configs={**comparison_configs(config),'custom':config}
        if reset_segment:
            self.segment+=1
        if not hasattr(self,'pipelines') or reset_segment:
            self.pipelines={name:Pipeline(c) for name,c in configs.items()}
        else:
            for name,c in configs.items():
                self.pipelines[name].set_config(c)

    def step(self, observation,t,aspect=1.,scenario='unlabelled',inference_ms=0.,truth=None):
        started=time.perf_counter()
        outputs={name:asdict(p.step(observation.points,t,aspect,observation.handedness,observation.handedness_score))
                 for name,p in self.pipelines.items()}
        post=(time.perf_counter()-started)*1000
        frame=dict(frame=self.frame,t=t,scenario=scenario,segment=self.segment,
                   raw=observation.points,world=observation.world,
                   handedness=observation.handedness,handedness_score=observation.handedness_score,
                   mediapipe_timestamp_ms=observation.timestamp_ms,
                   inference_ms=inference_ms,postprocess_ms=post,compute_ms=inference_ms+post,
                   config=self.config.to_dict(),outputs=outputs,truth=truth)
        self.frame+=1
        return frame
