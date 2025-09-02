#!/usr/bin/env python3
"""
NVIDIA GPU: 모델×배치 사이즈 성능/정확도 자동 측정 스위트

- weights/*.pt 를 모델로 자동 탐색 (또는 --models로 필터링)
- --dataset-root (기본: ./datasets/coco = annotations/ + val2017/)
- val.py의 run()을 직접 호출해 mAP/latency 확보, 총 벽시계 시간으로 overall_wall 계산
- OOM 나면 해당 배치부터 더 큰 배치는 스킵
- 결과를 마크다운 표로 콘솔/로그에 기록

사용 예:
  source venv/bin/activate
  python run_performance_suite_nvidia.py --device 1
  python run_performance_suite_nvidia.py --device 0 --batches 1,4,8 --models yolov9t,yolov8n
  python run_performance_suite_nvidia.py --dataset-root ../nvidia-yolo_250831/datasets/coco

주의:
- 이 스크립트는 현재 디렉토리(= val.py 있는 yolov9 루트)에서 실행한다고 가정.
- val.py 의 run() 시그니처와 반환 구조는 저장본을 그대로 사용( return_raw=True ).
"""

from __future__ import annotations
import argparse, os, re, sys, time, json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch

# --- YOLO repo 내부 모듈 import 준비 ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov9 root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from val import run as yolo_val_run  # :contentReference[oaicite:2]{index=2}

# -----------------------------
# 상수/경로
# -----------------------------
REPO_ROOT = ROOT
WEIGHTS_DIR_DEFAULT = REPO_ROOT / "weights"
DATASET_ROOT_DEFAULT = REPO_ROOT.parent / "datasets" / "coco"   # annotations/ + val2017/

# 실행 시점 기반 로그 파일명
START_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
FULL_LOG_FILE = REPO_ROOT / f"performance_full_nvidia_{START_TS}.log"    # 전체 로그
RESULT_LOG_FILE = REPO_ROOT / f"performance_result_nvidia_{START_TS}.log"  # 요약 로그

# COCO class names
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
    "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
    "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
    "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]
NC = 80

# 기본 임계값 (필요시 --target-map 으로 일괄 조정)
DEFAULT_TARGET_MAP = 0.3447
DEFAULT_CONF = 0.001
DEFAULT_IOU = 0.7
DEFAULT_IMGSZ = 640

# -----------------------------
# 로깅
# -----------------------------
def log_line(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with FULL_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def append_result(summary: str) -> None:
    try:
        with RESULT_LOG_FILE.open("a", encoding="utf-8") as rf:
            rf.write(summary + "\n\n")
    except Exception:
        pass

# -----------------------------
# 유틸
# -----------------------------
def discover_models(weights_dir: Path, models_filter: Optional[List[str]]) -> Dict[str, Path]:
    """weights_dir 안의 *.pt 탐색 -> {모델표시이름: 가중치경로}"""
    found = {}
    for p in sorted(weights_dir.glob("*.pt")):
        name = p.stem  # 예: yolov9-t-converted
        # 보기 좋게 표시용 이름을 살짝 정규화 (선택)
        disp = re.sub(r"[-_]?converted", "", name)
        disp = disp.replace("-", "")
        if models_filter and disp not in models_filter and name not in models_filter:
            continue
        found[disp] = p
    return found

def fmt(x):
    return "NA" if x is None else f"{x:.3f}"

# -----------------------------
# 실행(한 모델, 한 배치)
# -----------------------------
def run_one(weights_path: Path,
            batch_size: int,
            device: str,
            dataset_root: Path,
            imgsz: int,
            conf_thres: float,
            iou_thres: float,
            target_map: float) -> Optional[dict]:

    # val.py run() 에 넘길 data 딕셔너리 구성
    data = {
        "path": str(dataset_root),                                 # coco 루트
        "val": str(dataset_root / "val2017"),                      # 이미지 디렉토리
        "nc": NC,
        "names": {i: n for i, n in enumerate(COCO_CLASSES)}
    }

    # 성능 계측: 총 벽시계 시간
    t0 = time.time()
    try:
        # return_raw=True 로 호출하면 (mp, mr, map50, map, losses...), maps, (pre,inf,nms)ms, raw_lists 를 반환
        (mp, mr, map50, map95, *losses), maps, t_ms, raw = yolo_val_run(
            data=data,
            weights=str(weights_path),
            batch_size=batch_size,
            imgsz=imgsz,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            device=device,
            save_json=True,
            project=str(REPO_ROOT / "runs" / "val"),
            name=f"auto_{weights_path.stem}_bs{batch_size}_{START_TS}",
            exist_ok=True,
            half=True,
            return_raw=True,   # ← 커스텀 확장 (첨부본 val.py에 포함)  :contentReference[oaicite:3]{index=3}
            plots=False,
        )
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            log_line(f"[OOM] CUDA out of memory at bs={batch_size}. Skipping this and larger batches.")
            return {"oom": True}
        log_line(f"[ERROR] RuntimeError at bs={batch_size}: {e}")
        return None
    except Exception as e:
        log_line(f"[ERROR] Unexpected error at bs={batch_size}: {e}")
        return None

    wall_sec = time.time() - t0

    # per-image 평균(ms)
    pre_ms, inf_ms, nms_ms = t_ms  # tuples from val.py
    # e2e per-image(ms)
    e2e_ms = pre_ms + inf_ms + nms_ms

    # 처리 이미지 수 = raw list 길이(=개별 이미지 수)
    num_images = max(len(raw.get("pre", [])),
                     len(raw.get("inf", [])),
                     len(raw.get("nms", []))) or 0

    # throughput 계산
    infer_only_ips = 1000.0 / inf_ms if inf_ms else None
    e2e_active_ips = 1000.0 / e2e_ms if e2e_ms else None
    overall_wall_ips = (num_images / wall_sec) if num_images and wall_sec > 0 else None

    status = "success" if (map95 is not None and map95 >= target_map) else "failed"

    # warboy 스타일 JSON과 유사한 구조로 정리 (요약 표 생성에 맞춤)
    result = {
        "throughput_img_per_s": {
            "overall_wall": overall_wall_ips,
            "e2e_active": e2e_active_ips,
            "infer_only": infer_only_ips,
        },
        "latency_ms": {
            "pre":   {"avg": pre_ms},
            "infer": {"avg": inf_ms},
            "post":  {"avg": nms_ms},
            "e2e_active": {"avg": e2e_ms},
        },
        "dataset": {
            "throughput_wall_img_per_s": overall_wall_ips,
            "num_images": num_images,
        },
        "metrics": {
            "mAP": map95,
            "target": target_map,
            "status": status,
            "conf_thres": conf_thres,
            "iou_thres": iou_thres,
            "sec": wall_sec,
        }
    }

    # Compact 로그 한 줄
    log_line(
        "RESULT {} bs={} | overall_wall={} e2e_active={} infer_only={} | "
        "lat_pre_avg={} lat_infer_avg={} lat_post_avg={} | mAP={} target={} status={} | sec={}".format(
            weights_path.stem, batch_size,
            fmt(overall_wall_ips), fmt(e2e_active_ips), fmt(infer_only_ips),
            fmt(pre_ms), fmt(inf_ms), fmt(nms_ms),
            fmt(map95), fmt(target_map), status, fmt(wall_sec)
        )
    )

    # 원본 한 줄 JSON도 풀 로그에 저장
    try:
        with FULL_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "weights": str(weights_path),
                "batch_size": batch_size,
                "result": result
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass

    return result


# -----------------------------
# 요약 표
# -----------------------------
def summarize_by_model(model: str, by_bs: Dict[int, dict]) -> str:
    # 공통 conf/iou(첫 항목 기준)
    first = next(iter(by_bs.values()))
    metrics = first.get("metrics", {}) if first else {}
    conf_val = fmt(metrics.get("conf_thres"))
    iou_val  = fmt(metrics.get("iou_thres"))

    header = f"[model : {model}] : conf ({conf_val}), iou ({iou_val})"
    table = [
        "| batch_size | overall_wall | e2e_active | infer_only | lat_pre | lat_infer | lat_post | mAP | Target | Status | sec |",
        "|------------|--------------|------------|------------|---------|-----------|----------|-----|--------|--------|-----|",
    ]
    for bs in sorted(by_bs.keys()):
        r = by_bs[bs]
        if r is None or r.get("oom"):
            table.append(f"| {bs} | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM |")
            continue
        thr = r.get("throughput_img_per_s", {})
        lat = r.get("latency_ms", {})
        dataset = r.get("dataset", {})
        metrics = r.get("metrics", {})
        overall_wall = thr.get("overall_wall") or dataset.get("throughput_wall_img_per_s")
        row = f"| {bs} | {fmt(overall_wall)} | {fmt(thr.get('e2e_active'))} | {fmt(thr.get('infer_only'))} | " \
              f"{fmt(lat.get('pre',{}).get('avg'))} | {fmt(lat.get('infer',{}).get('avg'))} | {fmt(lat.get('post',{}).get('avg'))} | " \
              f"{fmt(metrics.get('mAP'))} | {fmt(metrics.get('target'))} | {metrics.get('status') or 'NA'} | {fmt(metrics.get('sec'))} |"
        table.append(row)
    return "\n".join([header, ""] + table + [""])

def summarize_by_batch(all_results: Dict[str, Dict[int, dict]], batch_size: int) -> str:
    header = f"[batch_size : {batch_size}]"
    table = [
        "| model | overall_wall | e2e_active | infer_only | lat_pre | lat_infer | lat_post | mAP | Target | Status | conf | iou | sec |",
        "|-------|--------------|------------|------------|---------|-----------|----------|-----|--------|--------|------|-----|-----|",
    ]
    for model, by_bs in all_results.items():
        if batch_size not in by_bs:
            continue
        r = by_bs[batch_size]
        if r is None or r.get("oom"):
            table.append(f"| {model} | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM |")
            continue
        thr = r.get("throughput_img_per_s", {})
        lat = r.get("latency_ms", {})
        dataset = r.get("dataset", {})
        metrics = r.get("metrics", {})
        overall_wall = thr.get("overall_wall") or dataset.get("throughput_wall_img_per_s")
        row = f"| {model} | {fmt(overall_wall)} | {fmt(thr.get('e2e_active'))} | {fmt(thr.get('infer_only'))} | " \
              f"{fmt(lat.get('pre',{}).get('avg'))} | {fmt(lat.get('infer',{}).get('avg'))} | {fmt(lat.get('post',{}).get('avg'))} | " \
              f"{fmt(metrics.get('mAP'))} | {fmt(metrics.get('target'))} | {metrics.get('status') or 'NA'} | " \
              f"{fmt(metrics.get('conf_thres'))} | {fmt(metrics.get('iou_thres'))} | {fmt(metrics.get('sec'))} |"
        table.append(row)
    return "\n".join([header, ""] + table + [""])


# -----------------------------
# 메인
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="NVIDIA GPU model performance suite (YOLO val.py 기반)")
    ap.add_argument("--device", default="0", help="CUDA device index, e.g., 0 or 1")
    ap.add_argument("--weights-dir", default=str(WEIGHTS_DIR_DEFAULT), help="weights directory (default: ./weights)")
    ap.add_argument("--dataset-root", default=str(DATASET_ROOT_DEFAULT), help="COCO root (annotations/ + val2017/)")
    ap.add_argument("--batches", default="1,4,8,16,32", help="comma-separated batch sizes to try")
    ap.add_argument("--models", default="", help="limit to these model names (comma-separated, match weight stem)")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="image size")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=DEFAULT_IOU, help="IoU threshold")
    ap.add_argument("--target-map", type=float, default=DEFAULT_TARGET_MAP, help="mAP50-95 target for pass/fail")
    args = ap.parse_args()

    # 시작 배너
    log_line("=" * 80)
    log_line(f"===== NVIDIA Suite started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    log_line(f"Full log file   : {FULL_LOG_FILE}")
    log_line(f"Result log file : {RESULT_LOG_FILE}")
    log_line("=" * 80)

    weights_dir = Path(args.weights_dir)
    dataset_root = Path(args.dataset_root)
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    models_filter = [m.strip() for m in args.models.split(",") if m.strip()] or None

    # 경로 체크
    if not weights_dir.exists():
        log_line(f"[ERROR] weights dir not found: {weights_dir}")
        sys.exit(1)
    if not (dataset_root / "annotations").exists() or not (dataset_root / "images" / "val2017").exists():
        log_line(f"[ERROR] dataset structure not found under: {dataset_root}")
        log_line(f"        expected: annotations/  and  images/val2017/")
        sys.exit(1)

    # 모델 탐색
    models = discover_models(weights_dir, models_filter)
    if not models:
        log_line(f"[ERROR] No *.pt weights found in {weights_dir}")
        sys.exit(1)

    log_line(f"Models to process: {list(models.keys())}")

    # 최종 결과 모음: all_results[model_name][bs] = result dict
    all_results: Dict[str, Dict[int, dict]] = {}

    for disp_name, wpath in models.items():
        log_line("=" * 80)
        log_line(f"PROCESS MODEL: {disp_name}")
        log_line(f"Available batches: {batches}")

        model_results: Dict[int, dict] = {}
        oom_seen = False
        for bs in batches:
            if oom_seen:
                log_line(f"[SKIP] previously OOM; skipping bs={bs} and larger for {disp_name}")
                model_results[bs] = {"oom": True}
                continue

            res = run_one(
                weights_path=wpath,
                batch_size=bs,
                device=args.device,
                dataset_root=dataset_root,
                imgsz=args.imgsz,
                conf_thres=args.conf,
                iou_thres=args.iou,
                target_map=args.target_map,
            )
            if res is None:
                log_line(f"[WARN] result missing for {disp_name} bs={bs}")
            else:
                model_results[bs] = res
                if res.get("oom"):
                    oom_seen = True

        all_results[disp_name] = model_results

        # --- 모델별 요약 (마크다운 표) ---
        block = summarize_by_model(disp_name, model_results)
        log_line("\n" + block + "\n")
        append_result(block)

    # --- 배치별 교차 요약 (마크다운 표) ---
    batch_sizes_present = sorted({bs for m in all_results.values() for bs in m.keys()})
    for bs in batch_sizes_present:
        block = summarize_by_batch(all_results, bs)
        log_line("\n" + block + "\n")

    log_line("All done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_line("Interrupted by user.")
        raise
