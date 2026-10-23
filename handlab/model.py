"""Explicit official model download; model binaries are excluded from Git."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'


def download_model(destination='models/hand_landmarker.task'):
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"기존 모델을 덮어쓰지 않습니다: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix('.part')
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, part.open('wb') as out:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 30 * 1024 * 1024:
                    raise ValueError('예상보다 큰 모델 다운로드입니다.')
                out.write(chunk)
        with zipfile.ZipFile(part) as z:
            if z.testzip() is not None or not any(n.endswith('.tflite') for n in z.namelist()):
                raise ValueError('유효한 MediaPipe task 번들이 아닙니다.')
        digest = hashlib.sha256(part.read_bytes()).hexdigest()
        part.replace(target)
        target.with_suffix('.json').write_text(json.dumps({'url': MODEL_URL, 'sha256': digest,
            'license': 'Apache-2.0', 'note': 'SHA256 records these bytes; not an independently verified publisher checksum.'}, indent=2), encoding='utf-8')
        return str(target), digest
    finally:
        part.unlink(missing_ok=True)
