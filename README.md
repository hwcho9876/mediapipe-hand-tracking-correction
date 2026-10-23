# HandLab: RGB 웹캠 손 모션캡처 보정 실험

일반 RGB 웹캠의 21개 손 좌표를 후처리하는 탐구용 Python 프로그램입니다. 공식 MediaPipe **Hand Landmarker Tasks API**를 사용하고 Legacy `mp.solutions.hands`를 사용하지 않습니다. CPU에서 작동하도록 설계했습니다.

## 설치

권장: Python 3.12, Windows/Linux x86-64. [Tasks Python 안내](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python), [지원 환경](https://developers.google.com/edge/mediapipe/solutions/setup_python), [PyPI](https://pypi.org/project/mediapipe/)를 확인하고 MediaPipe 1.1.1을 지정했습니다. 개발 환경에서 외부 패키지 다운로드가 차단되어 이 조합의 실제 추론은 아직 검증하지 못했습니다.

Linux에서 이 저장소의 루트 디렉터리에서 실행:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m handlab download-model
python -m handlab gui
```

Windows PowerShell에서 이 저장소의 루트 디렉터리에서 실행:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m handlab download-model
python -m handlab gui
```

활성화가 실행 정책으로 차단되면 `.\.venv\Scripts\python.exe -m pip install -e .` 및 `.\.venv\Scripts\python.exe -m handlab gui`를 사용하세요. Linux GUI에는 그래픽 데스크톱과 Tk 지원 Python이 필요합니다. 동영상만 처리하려면 GUI 없이 `video` 명령을 쓰세요. 카메라 입력 실패 시 OS 권한과 다른 앱이 카메라를 사용 중인지 확인하세요.

모델 다운로드는 명시적 명령으로만 실행됩니다. 공식 [모델 목록](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)의 float16 버전 1을 받아 `models/hand_landmarker.task`에 저장하며 바이너리를 Git에 올리지 않습니다. 받은 바이트의 SHA-256을 JSON 파일에 기록하지만 이는 게시자의 서명 검증값이 아닙니다. [모델 카드](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20%28Lite_Full%29%20with%20Fairness%20Oct%202021.pdf)와 MediaPipe 배포 라이선스는 Apache 2.0입니다. 외부 모델은 저장소에서 재배포하지 않습니다.

## 명령

```bash
python -m handlab gui --config config.default.json
python -m handlab video my_clip.mp4 --scenario fast
python -m handlab camera --device 0 --scenario stationary
python -m handlab synthetic --scenario occlusion --frames 240
python -m handlab analyze experiments/<session-name>
python -m unittest discover -s tests -v
```

GUI에서 `Start webcam`/`Open video`/`Stop`으로 입력을 관리합니다. 왼쪽은 파란 원본, 오른쪽은 선택한 보정 결과입니다. `View`에서 `raw`, `oef`, `corrected`, `full`, `custom`을 전환합니다. 네 알고리즘 스위치와 OEF/EMA 슬라이더는 실시간 적용됩니다. 바꾼 값은 새 `segment`로 저장하고 필터를 초기화하므로 다른 설정을 같은 구간으로 섞지 않습니다. `Scenario`는 사람이 실제로 수행한 동작에 붙이는 라벨입니다. `Record data`를 켜면 원본·5가지 출력·상태·설정·처리 시간을 JSONL 및 좌표 CSV로 저장합니다. 녹화를 끄거나 GUI를 닫으면 파일을 닫습니다. `analyze`는 `summary.json`과 `summary.csv`를 생성합니다. 저장 동영상을 같은 입력으로 다섯 경로에 전달하므로 조건을 비교할 수 있습니다.

## 모듈별 입력과 출력

| 모듈 | 입력 → 출력 | 역할 |
| --- | --- | --- |
| `capture`, `tracker` | 영상 프레임 → 21개 좌표·시각 | MediaPipe Tasks의 동기식 VIDEO CPU 추론 |
| `geometry`, `anomaly` | 새 좌표·기준 좌표 → 관절 마스크·근거 | 손 전체 이동과 관절 구조·가속도·화면 경계를 함께 판단 |
| `correction` | 믿을 수 있는 관절 이력 → 제한된 임시 좌표 | 일부 관절만 짧게 보정하고, 가짜 좌표를 새 관측으로 삼지 않음 |
| `filters` | 실제 관측·시간·관절 마스크 → OEF 출력 | 관절별 독립 상태. EMA 출력으로 내부 상태를 변경하지 않음 |
| `state`, `recovery`, `pipeline` | 상태·OEF 목표 → 최종 출력 | 일시 상실·EMA 재등장·OEF 복귀 |
| `experiment`, `recording`, `evaluation`, `render`, `gui` | 동일 관측 → 화면·로그·지표 | 5개 비교 경로와 실험 결과 저장 |

내부 좌표는 `(x, y × 화면세로/화면가로, z)`로, 각 축의 1은 이미지 가로 길이에 해당합니다. `z`는 손목 기준 상대 깊이 추정값이지 카메라까지의 실제 미터 거리가 아닙니다. MediaPipe가 반환하는 `hand_world_landmarks`는 별도로 기록하지만 이 보정에는 사용하지 않습니다. MediaPipe의 검출·존재·추적 confidence는 **설정 임계값**입니다. 출력 관절별 confidence는 이 API에서 얻지 못하므로 가정하지 않습니다. `handedness_score`는 왼손/오른손 분류 점수로만 사용합니다. 내부 추적 ROI도 공개 출력이 아니어서 화면 경계만 근사 지표로 씁니다.

OEF의 `min_cutoff`를 낮추면 천천히 움직일 때 떨림은 줄지만 지연이 늘 수 있습니다. `beta`를 높이면 빠른 동작의 지연을 줄이지만 노이즈를 통과시킬 수 있습니다. `d_cutoff`는 속도 추정을 매끄럽게 하는 정도입니다. `ema_tau`(초)를 줄이면 재등장 시 따라가는 속도가 빨라지지만 순간이동 억제가 약해집니다. 기타 매개변수는 [config.default.json](config.default.json)에 있으며 값은 합성 실험의 출발점이지 실측 최적값이 아닙니다.

`NORMAL`은 유효 추적, `SUSPECT`는 관절 오류 또는 모호한 이동, `LOST`는 추적 상실/임시 값 만료, `RECOVERING`은 재인식 후 EMA 출력입니다. 일관된 손 전체 점프는 3회 관측으로 확인합니다. 미검출 시 최대 `hold_seconds`만큼 제한된 값만 보여 줍니다. EMA는 같은 계수로 손 전체의 위치와 상대 손가락 자세를 추적하면서 OEF는 별개로 유효 관측을 계속 처리합니다. 두 출력 거리와 속도 차이를 확인해 전환하며, 복구가 길어지면 0.15초 출력 오프셋 축소 후 종료합니다. 너무 긴 상실이나 매우 먼 재등장은 명시적으로 즉시 재동기화하며 이때 불연속이 생길 수 있습니다.

정지 떨림, 프레임 점프, 재등장 불연속, 복구 시간, **원본 대비** 손목 x 상관 지연, 프레임 처리 시간과 FPS를 보고합니다. 실제 빠른 움직임도 점프 지표에 포함될 수 있고 `null` 지연은 신호/표본이 부족한 경우입니다. 일반 RGB 영상에는 정답 좌표가 없으므로 정확도 RMSE는 `null`; 출력 안정성과 정확도를 혼동하지 마세요. 합성 데이터는 인위적으로 만든 참좌표와 RMSE 비교가 가능합니다. 실제 정확도 검증에는 수동 라벨 또는 독립 기준 추적 좌표를 같은 시각·좌표계에 맞춰야 합니다. 모델이 자연스러운 형태의 잘못된 좌표를 계속 출력하면 후처리만으로 완벽히 찾을 수 없습니다. 현재 한 손만 지원합니다. 실제 카메라/GUI FPS와 오클루전 성능은 미검증입니다.

[처리 구조와 상태 정의](docs/ARCHITECTURE.md) · [단계별 테스트 기록](docs/DEVELOPMENT.md)
