"""Capture frame timing is separate from processing timing."""
import math
from pathlib import Path
import time


class VideoClock:
    def __init__(self, fps):
        self.fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        self.last = None
        self.fallbacks = 0

    def timestamp(self, pos_ms):
        t = pos_ms / 1000
        if not math.isfinite(t) or t < 0 or (self.last is not None and t <= self.last):
            t = 0.0 if self.last is None else self.last + 1 / self.fps
            self.fallbacks += 1
        self.last = t
        return t


class Capture:
    def __init__(self, source=0):
        import cv2
        self.cv2 = cv2
        self.live = isinstance(source, int)
        if not self.live and not Path(source).is_file():
            raise FileNotFoundError(f"동영상 없음: {source}")
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"입력을 열 수 없습니다: {source}. 카메라 권한/다른 앱 사용 여부를 확인하세요.")
        self.clock = VideoClock(self.cap.get(cv2.CAP_PROP_FPS))
        self.start = time.perf_counter()
        self.frame_index = 0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            if self.live:
                raise RuntimeError("카메라 프레임 읽기 실패: 연결과 권한을 확인하세요.")
            return None
        timestamp = time.perf_counter() - self.start if self.live else self.clock.timestamp(self.cap.get(self.cv2.CAP_PROP_POS_MSEC))
        self.frame_index += 1
        return timestamp, frame

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
