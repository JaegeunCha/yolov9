#!/usr/bin/env python3
"""
Extended NVIDIA E2E validation script.

- Default: run_e2e() once with given args
- --simple: run across all GPUs, all weights, batch sizes [in BATCH_SIZES]
            and summarize results in Markdown tables.

Notes
- Tables show per-image metrics:
  * e2e_wall_per_image (img/s): per-image wall throughput (현재는 e2e_active와 동일)
  * e2e_active (img/s): 1000 / avg(e2e_ms)
  * infer_only (img/s): 1000 / avg(infer_ms)
- Device header is written to both full and result logs.
"""

import argparse, json, os, sys, time, re
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import torch

from val import run as val_run
from benchmarks import run as bench_run

# -----------------------------
# Constants
# -----------------------------
REPO_ROOT = Path.cwd()
WEIGHT_DIR = REPO_ROOT / "weights"
START_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

FULL_LOG_FILE = LOG_DIR / f"nvidia_full_{START_TS}.log"
RESULT_LOG_FILE = LOG_DIR / f"nvidia_result_{START_TS}.log"

# 원하는 배치 조합으로 수정 가능
#BATCH_SIZES = [1, 4, 8, 16, 32]
BATCH_SIZES = [1]

TARGET_ACCURACY = {
    "yolov9t": 0.383,
    "yolov9s": 0.468,
    "yolov9m": 0.514,
    "yolov9c": 0.530,
    "yolov9e": 0.556,
    "yolov8n": 0.373,
    "yolov8l": 0.529,
}

# -----------------------------
# Metric Definitions
# -----------------------------
TRANSPOSED_METRICS = [
    #("e2e_wall_per_image", "e2e_wall (img/s)"),
    ("e2e_active", "e2e_active (img/s)"),
    ("infer_only", "infer_only (img/s)"),
    ("lat_pre", "lat_pre (ms)"),
    ("lat_infer", "lat_infer (ms)"),
    ("lat_post", "lat_post (ms)"),
    ("mAP", "mAP"),
    ("Target", "Target"),
    ("Status", "Status"),
    ("sec", "sec (s)"),
]


# -----------------------------
# Logging helpers
# -----------------------------
def log_line(msg: str, both: bool = True):
    if msg == "":  # 빈 문자열이면 그냥 개행
        if both:
            print("", flush=True)
        try:
            with FULL_LOG_FILE.open("a", encoding="utf-8") as f:
                f.write("\n")
        except Exception:
            pass
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if both:
        print(line, flush=True)
    try:
        with FULL_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# -----------------------------
# Utilities
# -----------------------------
def quantiles(arr):
    if not arr:
        return {"avg": None, "p50": None, "p90": None, "p99": None}
    arr_sorted = sorted(arr)
    n = len(arr_sorted)

    def pick(p):
        idx = max(0, min(n - 1, int(p * n) - 1))
        return arr_sorted[idx]

    return {
        "avg": sum(arr_sorted) / n,
        "p50": arr_sorted[n // 2],
        "p90": pick(0.90) if n >= 10 else None,
        "p99": pick(0.99) if n >= 100 else None,
    }


def run_e2e(opt):
    start = time.time()

    # 1) 마이크로 벤치(프리/인퍼/포스트 ms 리스트)
    lat_dict = bench_run(
        data=opt.data,
        weights=opt.weights,
        batch_size=opt.batch,
        imgsz=opt.img,
        device=opt.device,
        pt_only=True,
        half=opt.half,
    )
    e2e_ms_list = [p + i + n for p, i, n in zip(lat_dict["lat_pre"], lat_dict["lat_infer"], lat_dict["lat_post"])]

    # 2) 정확도(mAP) 측정
    results, maps, _ = val_run(
        data=opt.data,
        weights=opt.weights,
        batch_size=opt.batch,
        imgsz=opt.img,
        conf_thres=opt.conf,
        iou_thres=opt.iou,
        device=opt.device,
        save_json=True,
        name="e2e_val",
        half=opt.half,
        save_samples=getattr(opt, "save_samples", 0),
        sample_start=getattr(opt, "sample_start", 1),        
    )

    mAP = results[3]
    model_name = Path(opt.weights).stem.replace("-converted", "").replace("-", "")
    target = TARGET_ACCURACY.get(model_name, 0.3) * 0.9
    acc_check = "success" if mAP >= target else "fail"

    e2e_q = quantiles(e2e_ms_list)
    #wall_elapsed = time.time() - start
    #num_images = len(lat_dict["lat_pre"])
    #e2e_wall_imgps = num_images / wall_elapsed if wall_elapsed > 0 else None

    pre_q = quantiles(lat_dict["lat_pre"])
    inf_q = quantiles(lat_dict["lat_infer"])
    post_q = quantiles(lat_dict["lat_post"])

    acc_check = "success" if mAP >= target else "fail"

    summary = {
        "model": model_name,
        "images": len(lat_dict["lat_pre"]),
        "throughput_img_per_s": {
            #"e2e_wall_per_image": e2e_wall_imgps,
            "e2e_active": 1000.0 / e2e_q["avg"] if e2e_q["avg"] else None,
            "infer_only": 1000.0 / inf_q["avg"] if inf_q["avg"] else None
        },
        "latency_ms": {
            "pre": pre_q,
            "infer": inf_q,
            "post": post_q,
            "e2e_active": e2e_q,
        },
        "metrics": {
            "mAP": mAP,
            "target": target,
            "status": acc_check,
            "conf_thres": opt.conf,
            "iou_thres": opt.iou,
            "sec": time.time() - start,
        }
    }

    print(
        f"RESULT {model_name} bs={opt.batch} | "
        #f"e2e_wall_imgps={summary['throughput_img_per_s']['e2e_wall_per_image']} "        
        f"e2e_active={summary['throughput_img_per_s']['e2e_active']} "
        f"infer_only={summary['throughput_img_per_s']['infer_only']} | "
        f"lat_pre_avg={summary['latency_ms']['pre']['avg']} "
        f"lat_infer_avg={summary['latency_ms']['infer']['avg']} "
        f"lat_post_avg={summary['latency_ms']['post']['avg']} | "
        f"mAP={summary['metrics']['mAP']} target={summary['metrics']['target']} "
        f"status={summary['metrics']['status']} | "
        f"conf={summary['metrics']['conf_thres']} iou={summary['metrics']['iou_thres']} "
        f"sec={summary['metrics']['sec']:.2f}"
    )

    # 콘솔에도 JSON 한 번 출력(디버깅 가독성)
    print(json.dumps(summary, indent=2))

    return summary


def fmt(x):
    return "NA" if x is None else f"{x:.3f}"


def summarize_by_model(model: str, by_bs: Dict[int, dict]) -> str:
    first = next(iter(by_bs.values()))
    conf = fmt(first["metrics"].get("conf_thres"))
    iou = fmt(first["metrics"].get("iou_thres"))
    header = f"[model : {model}] : conf ({conf}), iou ({iou})"
    table = [
        #"| batch_size | e2e_wall_per_image (img/s) | e2e_active (img/s) | infer_only (img/s) | lat_pre (ms) | lat_infer (ms) | lat_post (ms) | mAP | Target | Status | sec (s) |",
        "| batch_size | e2e_active (img/s) | infer_only (img/s) | lat_pre (ms) | lat_infer (ms) | lat_post (ms) | mAP | Target | Status | sec (s) |",
        "|------------|--------------------|--------------------|--------------|----------------|---------------|-----|--------|--------|---------|",
    ]
    for bs in sorted(by_bs.keys()):
        r = by_bs[bs]
        thr = r["throughput_img_per_s"]
        lat = r["latency_ms"]
        acc = r["metrics"]
        table.append(
            #f"| {bs} | {fmt(thr.get('e2e_wall_per_image'))} | "
            f"| {bs} | "
            f"{fmt(thr.get('e2e_active'))} | {fmt(thr.get('infer_only'))} | "
            f"{fmt(lat['pre']['avg'])} | {fmt(lat['infer']['avg'])} | {fmt(lat['post']['avg'])} | "
            f"{fmt(acc['mAP'])} | {fmt(acc['target'])} | {acc['status']} | {fmt(acc['sec'])} |"
        )
    return "\n".join([header, ""] + table + [""])


def summarize_by_batch(batch: int, all_results: Dict[str, Dict[int, dict]]) -> str:
    first_model = next(iter(all_results.values()))
    first_res = first_model.get(batch, {})
    if first_res:
        conf_val = fmt(first_res["metrics"].get("conf_thres"))
        iou_val  = fmt(first_res["metrics"].get("iou_thres"))
    else:
        conf_val, iou_val = "NA", "NA"

    header = f"[batch_size : {batch}] : conf ({conf_val}), iou ({iou_val})"
    table = [
        #"| batch_size | e2e_wall_per_image (img/s) | e2e_active (img/s) | infer_only (img/s) | lat_pre (ms) | lat_infer (ms) | lat_post (ms) | mAP | Target | Status | sec (s) |",
        "| batch_size | e2e_active (img/s) | infer_only (img/s) | lat_pre (ms) | lat_infer (ms) | lat_post (ms) | mAP | Target | Status | sec (s) |",
        "|------------|--------------------|--------------------|--------------|----------------|---------------|-----|--------|--------|---------|",
    ]
    for model, bs_dict in all_results.items():
        if batch not in bs_dict:
            continue
        r = bs_dict[batch]
        thr = r["throughput_img_per_s"]
        lat = r["latency_ms"]
        acc = r["metrics"]
        table.append(
            #f"| {model} | {fmt(thr.get('e2e_wall_per_image'))} | "
            f"| {model} | "
            f"{fmt(thr.get('e2e_active'))} | {fmt(thr.get('infer_only'))} | "
            f"{fmt(lat['pre']['avg'])} | {fmt(lat['infer']['avg'])} | {fmt(lat['post']['avg'])} | "
            f"{fmt(acc['mAP'])} | {fmt(acc['target'])} | {acc['status']} | {fmt(acc['sec'])} |"
        )
    return "\n".join([header, ""] + table + [""])


def get_conf_iou(res: dict) -> tuple[str, str]:
    """metrics 블록에서 conf/iou 추출"""
    def fmt(x): return "NA" if x is None else f"{x:.3f}"
    m = res.get("metrics", {}) if res else {}
    return fmt(m.get("conf_thres")), fmt(m.get("iou_thres"))

def extract_metric_value(res: dict, metric: str):
    thr, lat, acc, comp = (
        res.get("throughput_img_per_s", {}),
        res.get("latency_ms", {}),
        res.get("metrics", {}),
        res.get("computed", {}),
    )
    #if metric == "e2e_wall_per_image":
    #    return thr.get("e2e_wall_per_image") or comp.get("e2e_wall_per_image")
    #elif metric == "e2e_active":
    if metric == "e2e_active":
        return thr.get("e2e") or thr.get("e2e_active")
    elif metric == "infer_only":
        return thr.get("infer_only")
    elif metric == "lat_pre":
        return lat.get("pre", {}).get("avg")
    elif metric == "lat_infer":
        return lat.get("infer", {}).get("avg")
    elif metric == "lat_post":
        return lat.get("post", {}).get("avg")
    elif metric == "mAP":
        return acc.get("mAP")
    elif metric == "Target":
        return acc.get("target")
    elif metric == "Status":
        return acc.get("status")
    elif metric == "conf":
        return acc.get("conf_thres")
    elif metric == "iou":
        return acc.get("iou_thres")
    elif metric == "sec":
        return acc.get("sec")
    return None

def summarize_transposed_by_model(model: str, by_bs: Dict[int, dict]) -> str:
    def fmt(x): return "NA" if x is None else f"{x:.3f}"
    batch_sizes = sorted(by_bs.keys())

    # conf/iou는 모든 batch에서 동일 → 첫 값 사용
    first_res = next(iter(by_bs.values()))
    conf_val, iou_val = get_conf_iou(first_res)

    rows = []
    for metric, label in TRANSPOSED_METRICS:
        row = [label]
        for bs in batch_sizes:
            res = by_bs[bs]
            val = extract_metric_value(res, metric)
            row.append(fmt(val) if metric != "Status" else (val or "NA"))
        rows.append("| " + " | ".join(row) + " |")


    header = ["batch_size"] + [str(bs) for bs in batch_sizes]
    table = [
        f"[transposed summary: model={model}] : conf ({conf_val}), iou ({iou_val})",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ] + rows
    return "\n".join(table) + "\n"


def summarize_transposed_by_batch(all_results: Dict[str, Dict[int, dict]], batch_size: int) -> str:
    def fmt(x): return "NA" if x is None else f"{x:.3f}"
    models = list(all_results.keys())

    # conf/iou는 모든 모델에서 동일하므로 첫 모델 값 사용
    first_model = next(iter(all_results.values()))
    first_res = first_model.get(batch_size, {})
    conf_val, iou_val = get_conf_iou(first_res)

    rows = []
    for metric, label in TRANSPOSED_METRICS:
        row = [label]
        for model in models:
            res = all_results[model].get(batch_size)
            if not res:
                row.append("NA")
                continue
            val = extract_metric_value(res, metric)
            row.append(fmt(val) if metric != "Status" else (val or "NA"))
        rows.append("| " + " | ".join(row) + " |")

    header = ["models"] + models
    table = [
        f"[transposed summary: batch_size={batch_size}] : conf ({conf_val}), iou ({iou_val})",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ] + rows
    return "\n".join(table) + "\n"



# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/coco.yaml")
    parser.add_argument("--weights", type=str, default="weights/yolov9t.pt")
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--conf", type=float, default=0.025)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument(
        "--save-samples",
        type=int,
        default=0,
        help="Save N sample prediction images with boxes+labels to outputs/ (0=disable)",
    )
    parser.add_argument(
        "--sample-start",
        type=int,
        default=1,
        help="Starting index (1-based) of dataset images to save with --save-samples",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Run all GPUs, all weights, batch sizes defined in BATCH_SIZES",
    )
    opt = parser.parse_args()

    if not opt.simple:
        summary = run_e2e(opt)
        print(json.dumps(summary, indent=2))
        return

    # --- Simple mode ---
    precision_str = "f16" if opt.half else "f32"
    log_line("=" * 80)
    log_line(f"===== Simple Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    log_line(f"Full log file: {FULL_LOG_FILE}")
    log_line(f"Result log file: {RESULT_LOG_FILE}")
    log_line("=" * 80)

    num_gpus = torch.cuda.device_count()
    weights = list(WEIGHT_DIR.glob("*.pt"))
    all_results: Dict[str, Dict[int, dict]] = {}

    #for dev_id in range(num_gpus):
    for dev_id in [1]:
        name = torch.cuda.get_device_name(dev_id)
        log_line("=" * 80)
        log_line(f"[device {dev_id} : {name}] [{precision_str}]", both=True)
        log_line("=" * 80)
        try:
            with RESULT_LOG_FILE.open("a", encoding="utf-8") as rf:
                rf.write("=" * 80 + "\n")
                rf.write(f"[device {dev_id} : {name}] [{precision_str}] \n")
                rf.write("=" * 80 + "\n")
        except Exception:
            pass
        
        for w in weights:
            model_name = w.stem
            log_line(f"PROCESS MODEL: {model_name}")
            model_results: Dict[int, dict] = {}

            for bs in BATCH_SIZES:
                class Opt:
                    pass

                o = Opt()
                o.data, o.weights, o.img = opt.data, str(w), opt.img
                o.batch, o.conf, o.iou = bs, opt.conf, opt.iou
                o.device, o.half = str(dev_id), opt.half
                o.save_samples = opt.save_samples
                o.sample_start = opt.sample_start

                try:
                    res = run_e2e(o)
                    model_results[bs] = res
                    log_line(
                        f"RESULT {model_name} bs={bs} | "
                        f"mAP={res['metrics']['mAP']} target={res['metrics']['target']} "
                        f"status={res['metrics']['status']} | "
                        f"conf={res['metrics']['conf_thres']} iou={res['metrics']['iou_thres']} "
                        f"sec={res['metrics']['sec']:.2f}"
                    )
                except Exception as e:
                    log_line(f"[WARN] Failed for {model_name} bs={bs}: {e}")

            if model_results:
                all_results[model_name] = model_results
                summary = summarize_by_model(model_name, model_results)
                log_line("\n" + summary)
                try:
                    with RESULT_LOG_FILE.open("a", encoding="utf-8") as rf:
                        rf.write(summary + "\n\n")
                except Exception:
                    pass
                print(summary)

                summary_t = summarize_transposed_by_model(model_name, model_results)
                log_line("\n" + summary_t)
                with RESULT_LOG_FILE.open("a", encoding="utf-8") as rf:
                    rf.write(summary_t + "\n\n")
                print(summary_t)

        # batch별 요약(모든 모델 누적 기준)
        batch_sizes_present = sorted({bs for m in all_results.values() for bs in m.keys()})
        for bs in batch_sizes_present:
            summary = summarize_by_batch(bs, all_results)
            log_line("\n" + summary)
            print(summary)
            try:
                with RESULT_LOG_FILE.open("a", encoding="utf-8") as rf:
                    rf.write(summary + "\n\n")
            except Exception:
                pass

            # transpose 요약 추가
            summary_t = summarize_transposed_by_batch(all_results, bs)
            log_line("\n" + summary_t)
            print(summary_t)
            try:
                with RESULT_LOG_FILE.open("a", encoding="utf-8") as rf:
                    rf.write(summary_t + "\n\n")
            except Exception:
                pass

    log_line("All done.")


if __name__ == "__main__":
    main()
