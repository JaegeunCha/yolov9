# yolov9 환경 설정 가이드

## 개요

이 문서는 `yolo-od-test` 프로젝트의 NVIDIA GPU 기반 YOLOv9 객체 탐지 환경을 구성하고 실행하는 방법을 설명합니다.
rsync 서버에 저장된 데이터를 다운로드하고, `yolov9`를 git clone으로 받아 환경을 완성합니다.

## 사전 요구사항

- rsync 서버(기본: `10.254.202.100`)에 SSH 접속이 가능해야 합니다
- `rsync`, `git`, `python3` 명령어가 설치되어 있어야 합니다
- `tree`는 스크립트가 자동으로 설치합니다
- NVIDIA GPU 및 CUDA 드라이버가 설치되어 있어야 합니다

## 설정 방법

### 1. 설정 스크립트 다운로드

rsync 서버에서 설정 스크립트를 다운로드합니다. (서버 IP는 환경에 맞게 변경)

```bash
scp kcloud@<서버IP>:~/data/setup_yolo_od_test.sh .
```

### 2. 스크립트 실행

```bash
chmod +x setup_yolo_od_test.sh

# 대화형으로 실행 (서버 IP 확인 → nvidia / furiosa / all 선택)
./setup_yolo_od_test.sh

# 또는 환경변수로 지정 (비대화형)
SERVER_A=10.254.202.100 SETUP_TARGET=nvidia ./setup_yolo_od_test.sh
```

스크립트가 수행하는 작업:
1. rsync 서버에서 선택한 대상의 데이터를 다운로드 (`yolov9`, `warboy-vision-models`, `venv`, 서버 전용 파일 제외)
2. `yolov9`를 git clone
3. `nvidia/models/weights/*.pt` → `nvidia/yolov9/weights/`로 weight 파일 복사
4. `nvidia/venv` Python 가상환경 신규 생성 및 패키지 설치

> **참고**: setuptools는 업그레이드하지 않습니다.
> 최신 setuptools(v78+)에서 `pkg_resources` 모듈이 제거되었으며,
> yolov9 코드가 `pkg_resources`를 사용하기 때문에 venv 기본 버전을 유지합니다.

### 3. 환경 변수

```bash
# 서버 IP (기본값: 10.254.202.100, 미지정 시 대화형으로 확인)
SERVER_A=10.254.202.100

# 설치 대상 (미지정 시 대화형으로 선택)
SETUP_TARGET=nvidia   # nvidia, furiosa, all

# SSH 사용자명 (기본값: kcloud)
SERVER_USER=myuser

# 로컬 저장 경로 (기본값: 현재 디렉토리)
LOCAL_BASE_DIR=/home/myuser/workspace
```

### 4. 추가 설치

이미 nvidia만 설치한 상태에서 furiosa를 추가할 수 있습니다:

```bash
SETUP_TARGET=furiosa ./setup_yolo_od_test.sh
```

## 실행 방법

### venv 활성화

모든 실행 명령은 nvidia venv를 활성화한 상태에서 수행해야 합니다.

```bash
source ~/yolo-od-test/nvidia/venv/bin/activate
cd ~/yolo-od-test/nvidia/yolov9
```

### nvidia_e2e_val.py — E2E 성능 평가

전체 GPU, 전체 weight에 대해 자동으로 성능 평가를 수행하고 결과를 Markdown 테이블로 요약합니다.

```bash
# 전체 자동 평가 (모든 GPU, 모든 weight, 정의된 batch size)
python3 nvidia_e2e_val.py --simple

# 특정 weight, batch size, GPU 지정
python3 nvidia_e2e_val.py --weights weights/yolov9t.pt --batch 32 --device 0

# half precision (FP16) 사용
python3 nvidia_e2e_val.py --weights weights/yolov9t.pt --batch 32 --device 0 --half

# 샘플 이미지 저장 (예: 10장, 1번째부터)
python3 nvidia_e2e_val.py --simple --save-samples 10 --sample-start 1
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--simple` | - | 모든 GPU/weight/batch 자동 순회 모드 |
| `--weights` | `weights/yolov9t.pt` | 모델 weight 파일 경로 |
| `--batch` | `32` | 배치 사이즈 |
| `--device` | `0` | GPU 번호 (예: `0`, `0,1`) |
| `--half` | - | FP16 반정밀도 사용 |
| `--data` | `data/coco.yaml` | 데이터셋 설정 파일 |
| `--conf` | `0.025` | confidence threshold |
| `--iou` | `0.7` | IoU threshold |
| `--save-samples` | `0` | 저장할 샘플 이미지 수 (0=비활성) |
| `--sample-start` | `None` | 샘플 저장 시작 인덱스 (1-based) |

결과 로그는 `logs/` 디렉토리에 저장됩니다:
- `nvidia_full_YYYYMMDD_HHMMSS.log` — 전체 실행 로그
- `nvidia_result_YYYYMMDD_HHMMSS.log` — 요약 결과 테이블

### venv 비활성화

작업이 끝나면 venv를 비활성화합니다.

```bash
deactivate
```

## 디렉토리 구조

### nvidia만 선택 시

```
yolo-od-test/
├── data/
│   └── setup_yolo_od_test.sh
├── dockerImage/
│   └── nvidia/
└── nvidia/
    ├── yolov9/                 ← git clone (이 저장소)
    │   ├── weights/            ← .pt 파일 (models/weights에서 복사됨)
    │   └── logs/               ← 실행 결과 로그
    ├── venv/                   ← 신규 생성
    ├── models/
    │   └── weights/            ← rsync로 다운로드된 원본 weight
    └── datasets/
```

### 모두 선택 시

```
yolo-od-test/
├── data/
│   └── setup_yolo_od_test.sh
├── dockerImage/
│   ├── furiosa/
│   └── nvidia/
├── furiosa/
│   ├── warboy-vision-models/   ← git clone
│   ├── venv/                   ← 신규 생성
│   ├── models/
│   └── datasets/
└── nvidia/
    ├── yolov9/                 ← git clone (이 저장소)
    ├── venv/                   ← 신규 생성
    ├── models/
    └── datasets/
```

## 참고

- `yolov9`는 NVIDIA GPU 기반 YOLOv9 객체 탐지 모델 저장소입니다
- 재실행 시 이미 clone된 저장소는 `git pull`로 업데이트됩니다
- venv는 매번 로컬에서 신규 생성되므로 서버 환경에 영향받지 않습니다
- rsync는 변경된 파일만 전송하므로 재실행 시에도 효율적입니다
- setuptools는 업그레이드하지 않습니다 (pkg_resources 호환성 유지)
