"""Validated JSON configuration, with no camera or GUI dependencies."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass
class Config:
    oef_enabled: bool = True
    detection_enabled: bool = True
    correction_enabled: bool = True
    ema_enabled: bool = True
    min_cutoff: float = 2.5
    beta: float = 5.0
    d_cutoff: float = 1.0
    ema_tau: float = 0.09
    recovery_timeout: float = 0.7
    recovery_distance: float = 0.035
    recovery_velocity: float = 1.0
    recovery_confirm: int = 3
    resync_distance: float = 4.0
    long_gap: float = 1.0
    prediction_seconds: float = 0.08
    prediction_limit: float = 0.35
    hold_seconds: float = 0.25
    suspect_confirm: int = 3
    jump_distance: float = 0.7
    jump_speed: float = 12.0
    acceleration: float = 180.0
    joint_residual: float = 0.55
    bone_ratio: float = 2.5
    screen_margin: float = 0.15

    def validate(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if f.type is bool:
                if not isinstance(v, bool):
                    raise ValueError(f"{f.name}: true/false가 필요합니다.")
            elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{f.name}: 유한한 숫자가 필요합니다.")
            elif f.type is int and (not isinstance(v, int) or not 1 <= v <= 30):
                raise ValueError(f"{f.name}: 1~30 정수가 필요합니다.")
            elif f.type is float and (v < 0 or (v == 0 and f.name not in ('beta', 'screen_margin'))):
                raise ValueError(f"{f.name}: 양수가 필요합니다.")
        if self.bone_ratio <= 1:
            raise ValueError("bone_ratio는 1보다 커야 합니다.")
        if self.prediction_seconds > self.hold_seconds or self.hold_seconds > self.long_gap:
            raise ValueError("prediction_seconds <= hold_seconds <= long_gap이어야 합니다.")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"알 수 없는 설정: {sorted(unknown)}")
        return cls(**data).validate()

    @classmethod
    def load(cls, path=None):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8"))) if path else cls().validate()

    def save(self, path):
        self.validate()
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
