"""CPU Hand Landmarker Tasks adapter. Heavy imports occur only when instantiated."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .geometry import to_width_coordinates


@dataclass
class Observation:
    points: np.ndarray | None
    world: np.ndarray | None = None
    handedness: str | None = None
    handedness_score: float | None = None
    timestamp_ms: int | None = None


def convert_result(result, width, height, timestamp_ms):
    if not result.hand_landmarks:
        return Observation(None, timestamp_ms=timestamp_ms)
    landmarks = result.hand_landmarks[0]
    points = to_width_coordinates([[p.x, p.y, p.z] for p in landmarks], width, height)
    world = None
    if result.hand_world_landmarks:
        world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]])
    category = result.handedness[0][0] if result.handedness and result.handedness[0] else None
    return Observation(points, world, category.category_name if category else None,
                       float(category.score) if category else None, timestamp_ms)


class HandTracker:
    def __init__(self, model_path, detection=0.5, presence=0.5, tracking=0.5):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"모델 파일 없음: {model_path}. python -m handlab download-model 실행")
        if not all(0 <= c <= 1 for c in (detection, presence, tracking)):
            raise ValueError("MediaPipe confidence 임계값은 0~1이어야 합니다.")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
        self.mp = mp
        # Synchronous VIDEO also accepts sequential webcam frames. No dropped async callbacks.
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.CPU),
            running_mode=RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=detection, min_hand_presence_confidence=presence,
            min_tracking_confidence=tracking)
        self.task = HandLandmarker.create_from_options(options)
        self.last_ms = -1

    def detect(self, bgr, timestamp):
        import cv2
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("타임스탬프는 0 이상의 유한한 초 단위 값이어야 합니다.")
        ms = max(self.last_ms + 1, int(round(timestamp * 1000)))
        self.last_ms = ms
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self.task.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ms)
        h, w = bgr.shape[:2]
        return convert_result(result, w, h, ms)

    def close(self):
        self.task.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
