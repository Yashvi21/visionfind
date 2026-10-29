"""Evaluate VisionFind indexing + visual search at different sampling intervals.

Reports, per interval: frame count, indexing time, index size, Recall@5, MRR@10 and search latency.

Usage (from ml/, with the venv active):
    python scripts/evaluate.py                         # built-in synthetic benchmark (generated video)
    python scripts/evaluate.py --eval-set my_set.json  # your own labelled videos
    python scripts/evaluate.py --intervals 2 5 --repeats 10 --out report.json

Eval-set format (see scripts/eval_set.example.json):
    {"videos": [{"path": "/abs/or/relative/video.mp4",
                 "queries": [{"query": "a person at a whiteboard", "relevant": [[12.0, 30.5], [95, 110]]}]}]}
A query counts as a hit when a returned frame's timestamp lies inside any of its [start, end] ranges.

Metrics:
    Recall@5  share of queries with at least one relevant frame in the top 5 results
    MRR@10    mean of 1/rank of the first relevant result in the top 10 (0 if none)
    latency   end-to-end search time per query (CLIP text embedding + FAISS + SQLite), median and p95
Search uses the same code path as the app (`search_frames`, including temporal dedupe), restricted to
the video each query belongs to.

Everything runs in a temporary data directory; your real library and indexes are never touched.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
import uuid
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--eval-set", type=Path, help="JSON file with labelled videos (default: synthetic benchmark)")
    p.add_argument("--intervals", type=int, nargs="+", default=[1, 2, 5], choices=[1, 2, 5])
    p.add_argument("--repeats", type=int, default=5, help="timed runs per query for latency (default 5)")
    p.add_argument("--out", type=Path, help="also write the full report as JSON")
    p.add_argument("--keep-data", action="store_true", help="keep the temporary data directory")
    return p.parse_args()


ARGS = parse_args()
TMP = Path(tempfile.mkdtemp(prefix="visionfind-eval-"))
# Must be set before any app module is imported (config reads them at import time).
os.environ["VS_DATA_DIR"] = str(TMP / "data")
os.environ["VS_PRELOAD_MODELS"] = "0"
sys.path.insert(0, str(ML_ROOT))

import logging  # noqa: E402

import numpy as np  # noqa: E402  (imported after app's OpenMP setup below)

import app  # noqa: E402,F401  sets OMP_NUM_THREADS before torch/faiss load
from app import config  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.embeddings import ClipEmbedder  # noqa: E402
from app.models import Video  # noqa: E402
from app.search import search_frames  # noqa: E402
from app.services import Services  # noqa: E402
from app.transcribe import MiniLMEmbedder, WhisperTranscriber  # noqa: E402
from app.video_io import probe  # noqa: E402
from app.worker import Worker  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Synthetic benchmark
# ---------------------------------------------------------------------------
def _scene_painters():
    import cv2

    W, H = 960, 540

    def bg(color):
        return np.full((H, W, 3), color, np.uint8)

    def red_circle(i):
        f = bg((255, 255, 255)); cv2.circle(f, (W // 2 + (i % 20) - 10, H // 2), 170, (30, 30, 220), -1); return f

    def blue_square(i):
        f = bg((60, 220, 250)); cv2.rectangle(f, (300 + i % 15, 120), (660 + i % 15, 420), (200, 80, 20), -1); return f

    def green_triangle(i):
        f = bg((240, 240, 240)); pts = np.array([[480, 90 + i % 10], [220, 450], [740, 450]], np.int32)
        cv2.fillPoly(f, [pts], (40, 170, 40)); return f

    def checkerboard(i):
        f = bg((0, 0, 0)); s = 60
        for y in range(0, H, s):
            for x in range(0, W, s):
                if ((x // s) + (y // s) + (i // 15)) % 2:
                    f[y:y + s, x:x + s] = 255
        return f

    def countryside(i):
        f = bg((0, 0, 0)); f[: H // 2] = (235, 180, 100); f[H // 2:] = (40, 160, 40)
        cv2.circle(f, (760 + i % 20, 110), 60, (0, 220, 255), -1); return f

    def whiteboard(i):
        f = bg((245, 245, 245)); cv2.rectangle(f, (20, 20), (W - 20, H - 20), (90, 90, 90), 8)
        for k, t in enumerate(["Q3 ROADMAP", "- launch beta", "- hire 2 devs"]):
            cv2.putText(f, t, (80, 150 + k * 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (160, 40, 20), 5)
        return f

    def night_moon(i):
        f = bg((40, 15, 10)); rng = np.random.default_rng(7)
        for x, y in rng.integers(0, [W, H], size=(120, 2)):
            cv2.circle(f, (int(x), int(y)), 2, (255, 255, 255), -1)
        cv2.circle(f, (700, 150), 80, (210, 235, 245), -1); return f

    def brick_wall(i):
        f = bg((60, 60, 60)); bw, bh = 120, 50
        for r, y in enumerate(range(0, H, bh)):
            off = (bw // 2) * (r % 2)
            for x in range(-off, W, bw):
                cv2.rectangle(f, (x + 3, y + 3), (x + bw - 3, y + bh - 3), (40, 70, 170), -1)
        return f

    def beach(i):
        f = bg((0, 0, 0)); f[: H // 3] = (235, 200, 140); f[H // 3: 2 * H // 3] = (180, 110, 20); f[2 * H // 3:] = (140, 200, 230)
        for k in range(6):
            y = H // 3 + 25 * k + (i % 10)
            cv2.line(f, (0, y), (W, y + 8), (240, 230, 220), 3)
        return f

    def stop_sign(i):
        f = bg((200, 200, 200)); c, r = (W // 2, H // 2), 200
        pts = np.array([[c[0] + r * np.cos(np.pi / 8 + k * np.pi / 4), c[1] + r * np.sin(np.pi / 8 + k * np.pi / 4)] for k in range(8)], np.int32)
        cv2.fillPoly(f, [pts], (30, 30, 200))
        cv2.putText(f, "STOP", (c[0] - 150, c[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (255, 255, 255), 10); return f

    def bar_chart(i):
        f = bg((255, 255, 255)); cv2.line(f, (100, 460), (880, 460), (0, 0, 0), 3); cv2.line(f, (100, 460), (100, 60), (0, 0, 0), 3)
        for k, h in enumerate([120, 250, 180, 330, 290]):
            cv2.rectangle(f, (150 + k * 140, 460 - h), (230 + k * 140, 460), (180, 120, 40), -1)
        return f

    def clock(i):
        f = bg((230, 230, 230)); c = (W // 2, H // 2)
        cv2.circle(f, c, 220, (255, 255, 255), -1); cv2.circle(f, c, 220, (0, 0, 0), 6)
        for k in range(12):
            a = k * np.pi / 6
            cv2.line(f, (int(c[0] + 190 * np.sin(a)), int(c[1] - 190 * np.cos(a))), (int(c[0] + 210 * np.sin(a)), int(c[1] - 210 * np.cos(a))), (0, 0, 0), 5)
        cv2.line(f, c, (c[0] + 110, c[1] - 60), (0, 0, 0), 9); cv2.line(f, c, (c[0] - 20, c[1] - 170), (0, 0, 0), 5); return f

    return {
        "red_circle": red_circle, "blue_square": blue_square, "green_triangle": green_triangle,
        "checkerboard": checkerboard, "countryside": countryside, "whiteboard": whiteboard,
        "night_moon": night_moon, "brick_wall": brick_wall, "beach": beach, "stop_sign": stop_sign,
        "bar_chart": bar_chart, "clock": clock,
    }


# Queries are worded differently from how the scenes were drawn, as a user would phrase them.
SYNTHETIC_QUERIES = {
    "red_circle": "a red circle",
    "blue_square": "a blue square on a yellow background",
    "green_triangle": "a green triangle",
    "checkerboard": "a chessboard pattern",
    "countryside": "a sunny field with blue sky",
    "whiteboard": "handwritten notes on a whiteboard",
    "night_moon": "the moon and stars at night",
    "brick_wall": "a red brick wall",
    "beach": "a beach with the sea",
    "stop_sign": "a stop sign",
    "bar_chart": "a bar chart",
    "clock": "a clock face",
}
# (scene, seconds). Two scenes appear twice, so those queries have two relevant ranges.
SYNTHETIC_TIMELINE = [
    ("whiteboard", 12), ("countryside", 9), ("red_circle", 7), ("bar_chart", 11), ("night_moon", 8),
    ("checkerboard", 6), ("brick_wall", 10), ("beach", 9), ("stop_sign", 7), ("green_triangle", 8),
    ("clock", 9), ("blue_square", 10), ("whiteboard", 8), ("checkerboard", 7),
]


def build_synthetic(dest: Path) -> dict:
    import cv2

    painters, fps = _scene_painters(), 10
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))
    ranges: dict[str, list[list[float]]] = {}
    t = 0.0
    for scene, secs in SYNTHETIC_TIMELINE:
        for i in range(secs * fps):
            writer.write(painters[scene](i))
        ranges.setdefault(scene, []).append([t, t + secs])
        t += secs
    writer.release()
    return {"videos": [{"path": str(dest), "queries": [{"query": SYNTHETIC_QUERIES[s], "relevant": r} for s, r in ranges.items()]}]}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def reset_data_dir() -> None:
    with SessionLocal() as s:
        for v in s.query(Video).all():
            s.delete(v)
        s.commit()
    for d in (config.VIDEOS_DIR, config.FRAMES_DIR, config.INDEX_DIR):
        shutil.rmtree(d, ignore_errors=True)
    config.ensure_dirs()


def is_relevant(t: float, ranges: list[list[float]]) -> bool:
    return any(a <= t <= b for a, b in ranges)


def evaluate_interval(interval: int, eval_set: dict, models: dict) -> dict:
    reset_data_dir()
    services = Services(
        clip_factory=lambda: models["clip"],
        text_factory=lambda: models["text"],
        transcriber_factory=lambda: models["whisper"],
    )
    worker = Worker(services)

    video_ids, frames_total, index_secs, duration_total = [], 0, 0.0, 0.0
    for v in eval_set["videos"]:
        src = Path(v["path"]).expanduser()
        vid = str(uuid.uuid4())
        dest = config.VIDEOS_DIR / f"{vid}.mp4"
        shutil.copyfile(src, dest)
        info = probe(dest)
        with SessionLocal() as s:
            s.add(Video(id=vid, filename=src.name, stored_path=str(dest.relative_to(config.DATA_DIR)),
                        duration_sec=info.duration_sec, size_bytes=dest.stat().st_size, sample_interval_sec=interval))
            s.commit()
        started = time.perf_counter()
        worker.process(vid)  # the exact pipeline the background worker runs
        index_secs += time.perf_counter() - started
        with SessionLocal() as s:
            video = s.get(Video, vid)
            if video.status != "ready":
                raise SystemExit(f"Indexing failed for {src}: {video.error_message}")
            frames_total += video.frame_count
        duration_total += info.duration_sec
        video_ids.append(vid)

    index_bytes = dir_size(config.INDEX_DIR)
    frame_bytes = dir_size(config.FRAMES_DIR)

    recall_hits, rr, latencies, per_query = 0, [], [], []
    with SessionLocal() as db:
        for vid, v in zip(video_ids, eval_set["videos"]):
            for q in v["queries"]:
                search_frames(services, db, models["clip"].encode_text([q["query"]]), 10, vid)  # untimed warm-up
                times = []
                for _ in range(max(1, ARGS.repeats)):
                    t0 = time.perf_counter()
                    hits = search_frames(services, db, models["clip"].encode_text([q["query"]]), 10, vid)
                    times.append((time.perf_counter() - t0) * 1000)
                latencies.extend(times)
                ranks = [i + 1 for i, h in enumerate(hits) if is_relevant(h.timestamp_sec, q["relevant"])]
                first = ranks[0] if ranks else None
                recall_hits += bool(first and first <= 5)
                rr.append(1 / first if first else 0.0)
                per_query.append({"query": q["query"], "first_relevant_rank": first,
                                  "top5": [round(h.timestamp_sec, 1) for h in hits[:5]]})

    n = len(rr)
    lat_sorted = sorted(latencies)
    return {
        "interval_sec": interval,
        "videos": len(video_ids),
        "video_minutes": round(duration_total / 60, 2),
        "frames": frames_total,
        "index_time_sec": round(index_secs, 2),
        "index_sec_per_video_min": round(index_secs / (duration_total / 60), 2) if duration_total else None,
        "faiss_index_bytes": index_bytes,
        "frame_jpeg_bytes": frame_bytes,
        "queries": n,
        "recall_at_5": round(recall_hits / n, 3) if n else None,
        "mrr_at_10": round(sum(rr) / n, 3) if n else None,
        "search_ms_p50": round(statistics.median(lat_sorted), 1),
        "search_ms_p95": round(lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))], 1),
        "per_query": per_query,
    }


def fmt_bytes(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"


def main() -> None:
    init_db()
    if ARGS.eval_set:
        eval_set = json.loads(ARGS.eval_set.read_text())
        base = ARGS.eval_set.resolve().parent
        for v in eval_set["videos"]:
            p = Path(v["path"]).expanduser()
            v["path"] = str(p if p.is_absolute() else base / p)
            if not Path(v["path"]).is_file():
                raise SystemExit(f"Video not found: {v['path']}")
        source = str(ARGS.eval_set)
    else:
        print("Generating synthetic benchmark video…")
        eval_set = build_synthetic(TMP / "synthetic_benchmark.mp4")
        source = "synthetic benchmark (14 scenes, 12 queries, 2 min)"

    print("Loading models (CLIP, MiniLM, Whisper)…")
    models = {"clip": ClipEmbedder(), "text": MiniLMEmbedder(), "whisper": WhisperTranscriber()}
    # The first MPS/CUDA call for each model path is slow; keep it out of the timings.
    import cv2

    warm = TMP / "warmup.jpg"
    cv2.imwrite(str(warm), np.zeros((224, 224, 3), np.uint8))
    models["clip"].encode_images([warm] * 4)
    models["clip"].encode_text(["warm up"])
    models["text"].encode(["warm up"])

    results = []
    for interval in ARGS.intervals:
        print(f"Indexing + searching at {interval}s interval…")
        results.append(evaluate_interval(interval, eval_set, models))

    device = getattr(models["clip"], "device", "?")
    print(f"\nVisionFind evaluation: {source} | CLIP on {device} | latency over {ARGS.repeats} runs/query\n")
    header = f"{'interval':>8} {'frames':>7} {'index time':>11} {'s / video min':>14} {'FAISS index':>12} {'frame JPEGs':>12} {'Recall@5':>9} {'MRR@10':>7} {'search p50':>11} {'p95':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{str(r['interval_sec']) + 's':>8} {r['frames']:>7} {r['index_time_sec']:>10.1f}s {r['index_sec_per_video_min']:>14.1f} "
              f"{fmt_bytes(r['faiss_index_bytes']):>12} {fmt_bytes(r['frame_jpeg_bytes']):>12} {r['recall_at_5']:>9.2f} "
              f"{r['mrr_at_10']:>7.2f} {r['search_ms_p50']:>9.1f}ms {r['search_ms_p95']:>6.1f}ms")

    misses = {q["query"] for r in results for q in r["per_query"] if not q["first_relevant_rank"] or q["first_relevant_rank"] > 5}
    if misses:
        print("\nQueries without a relevant frame in the top 5 (at some interval):")
        for q in sorted(misses):
            print(f"  - {q}")
    if not ARGS.eval_set:
        print("\nNote: the synthetic scenes are simple and visually distinct, so these accuracy numbers are an upper bound."
              "\nUse --eval-set with your own labelled videos for realistic Recall/MRR.")

    if ARGS.out:
        ARGS.out.write_text(json.dumps({"source": source, "device": device, "results": results}, indent=2))
        print(f"\nFull report written to {ARGS.out}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if ARGS.keep_data:
            print(f"Data kept in {TMP}")
        else:
            shutil.rmtree(TMP, ignore_errors=True)
