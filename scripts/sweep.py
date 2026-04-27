"""Orchestrate an N-ablation × M-seed sweep across an arbitrary number of
CUDA devices (or MIG instances).

Each (ablation, seed) becomes one `geompnn.train` invocation pinned to a single
device via CUDA_VISIBLE_DEVICES. We launch one such process per slot and
rotate through the queue.

Slots are flexible: pass either integer GPU indices (`--gpus 0 1`) or MIG UUIDs
(`--gpus MIG-abc... MIG-def...`). Each value is forwarded verbatim as
CUDA_VISIBLE_DEVICES, so anything `nvidia-smi` accepts works.

The script emits a **FINAL ETA** on every completion event and every
`--eta_interval` seconds. ETA is computed from the rolling mean per-run wall
time observed so far; before any run completes, it falls back to the
`--eta_seed_hours` seed value (default 9 h, our measured A40 baseline).

Usage:
    # Smoke (5 epochs, 8 train sims, no eval) for one ablation
    python scripts/sweep.py --ablations s2v --seeds 0 \
        --override nb_epochs=5 subsampling=2000 --limit_train 8 --skip_eval

    # Full reproduction across all detected MIG slots
    python scripts/sweep.py \
        --ablations mlp gnn s2v s2v_gnn trail polar sine sph inlet geompnn \
        --seeds 0 1 2 3 4 5 6 7 \
        --gpus MIG-uuid-1 MIG-uuid-2 ... MIG-uuid-N

Run state is persisted under results/<ablation>/seed<k>/. Re-invoking the
script skips any (ablation, seed) that already has score.json (or, with
--skip_eval, ckpt.pt) — so a Ctrl-C and restart resumes cleanly.
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "src" / "geompnn" / "configs"
RESULTS_DIR = ROOT / "results"
SRC_DIR = ROOT / "src"


def queue(ablations, seeds):
    return [(ab, s) for ab in ablations for s in seeds]


def already_done(ab: str, seed: int) -> bool:
    return (RESULTS_DIR / ab / f"seed{seed}" / "score.json").exists()


def already_skip_eval_done(ab: str, seed: int) -> bool:
    return (RESULTS_DIR / ab / f"seed{seed}" / "ckpt.pt").exists()


def fmt_dur(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # negative or NaN
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def fmt_eta_clock(eta_seconds: float) -> str:
    if eta_seconds < 0 or eta_seconds != eta_seconds:
        return "?"
    eta_dt = datetime.now() + timedelta(seconds=eta_seconds)
    return eta_dt.strftime("%Y-%m-%d %H:%M:%S")


def estimate_eta(durations: list[float], slots: int, remaining: int,
                 in_flight: int, default_seed: float) -> tuple[float, str]:
    """Return (eta_seconds, source_label).

    Each completed run is *one* (ablation, seed) end-to-end run. We treat slots
    as a fixed parallel pool and compute:
        eta = ceil((remaining + in_flight) / slots) * mean_run_time

    `in_flight` is included because those processes also need to finish.
    """
    if not durations:
        per_run = default_seed * 3600.0
        source = f"default {default_seed:.1f}h/run"
    else:
        per_run = statistics.mean(durations)
        source = f"observed mean over {len(durations)} run(s)"
    if slots <= 0:
        return float("inf"), source
    pending_runs = remaining + in_flight
    cycles = -(-pending_runs // slots)  # ceil div
    eta = cycles * per_run
    return eta, source


def launch(ab: str, seed: int, gpu: str, args, env_extra=None) -> subprocess.Popen:
    out_dir = RESULTS_DIR / ab / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG_DIR / f"{ab}.json"
    if not cfg.exists():
        raise FileNotFoundError(f"Missing ablation config {cfg}")

    cmd = [
        sys.executable, "-m", "geompnn.train",
        "--config", str(cfg),
        "--seed", str(seed),
        "--output_dir", str(out_dir),
    ]
    if args.limit_train is not None:
        cmd += ["--limit_train_simulations", str(args.limit_train)]
    if args.skip_eval:
        cmd += ["--skip_eval"]
    if args.override:
        cmd += ["--override", *args.override]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(SRC_DIR) + ":" + env.get("PYTHONPATH", "")
    if env_extra:
        env.update(env_extra)

    log_path = out_dir / "stdout.log"
    log_f = open(log_path, "w")
    print(f"[sweep] launch {ab}/seed{seed} on dev={gpu}  (-> {out_dir.relative_to(ROOT)})")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    proc._log_file = log_f
    proc._tag = f"{ab}/seed{seed}"
    proc._gpu = gpu
    proc._start = time.time()
    return proc


def report(eta_source: str, completed: int, total: int, in_flight: int,
           remaining: int, durations: list[float], slots: int,
           default_seed: float, prefix: str = "ETA"):
    eta_seconds, src = estimate_eta(durations, slots, remaining, in_flight, default_seed)
    eta_clock = fmt_eta_clock(eta_seconds)
    pace = (
        f"avg/run={fmt_dur(statistics.mean(durations))}"
        if durations else
        f"avg/run=<{default_seed:.1f}h placeholder>"
    )
    msg = (
        f"[sweep] {prefix}: done={completed}/{total} in_flight={in_flight} "
        f"remaining={remaining} slots={slots} {pace} | source: {src} | "
        f"FINAL ETA = {fmt_dur(eta_seconds)} (~{eta_clock})"
    )
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)
    parser.add_argument("--ablations", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--gpus", nargs="+", required=True,
                        help="GPU device identifiers (integer indices OR MIG UUIDs). "
                             "One concurrent training per identifier.")
    parser.add_argument("--override", nargs="+", default=None,
                        help="Inline overrides forwarded to geompnn.train")
    parser.add_argument("--limit_train", type=int, default=None,
                        help="Truncate train_dataset (smoke runs).")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip scoring (used for fast smoke runs).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if score.json/ckpt.pt already exists.")
    parser.add_argument("--eta_interval", type=int, default=300,
                        help="Seconds between heartbeat ETA reports (default 300).")
    parser.add_argument("--eta_seed_hours", type=float, default=9.0,
                        help="Per-run hours to use for ETA before any run completes "
                             "(default 9.0, our measured A40 baseline).")
    args = parser.parse_args()

    work_full = queue(args.ablations, args.seeds)
    work = list(work_full)
    skipped = 0
    if not args.force:
        before = len(work)
        if args.skip_eval:
            work = [(a, s) for (a, s) in work if not already_skip_eval_done(a, s)]
        else:
            work = [(a, s) for (a, s) in work if not already_done(a, s)]
        skipped = before - len(work)

    total = len(work_full)
    completed = skipped
    print(f"[sweep] total runs={total}  skipped={skipped}  to-run={len(work)}")
    print(f"[sweep] slots={len(args.gpus)}: {args.gpus}")

    durations: list[float] = []
    running: dict[str, subprocess.Popen] = {}
    failures: list[tuple[str, int]] = []
    last_report = time.time()

    # Initial ETA report (before any completions, uses default seed)
    report("seed", completed, total, 0, len(work), durations, len(args.gpus),
           args.eta_seed_hours, prefix="INITIAL ETA")

    while work or running:
        # Launch on free slots
        free_slots = [g for g in args.gpus if g not in running]
        for gpu in free_slots:
            if not work:
                break
            ab, seed = work.pop(0)
            running[gpu] = launch(ab, seed, gpu, args)

        time.sleep(5)
        finished_devs = []
        for gpu, proc in running.items():
            ret = proc.poll()
            if ret is None:
                continue
            proc._log_file.close()
            elapsed = time.time() - proc._start
            tag = proc._tag
            if ret == 0:
                durations.append(elapsed)
                completed += 1
                print(f"[sweep] DONE  {tag} on dev={gpu}  (took {fmt_dur(elapsed)})")
            else:
                print(f"[sweep] FAIL  {tag} on dev={gpu}  rc={ret}  ({fmt_dur(elapsed)})")
                failures.append((tag, ret))
            finished_devs.append(gpu)

        for gpu in finished_devs:
            del running[gpu]

        # Report ETA on every completion AND every eta_interval seconds
        if finished_devs or (time.time() - last_report >= args.eta_interval):
            report("rolling", completed, total, len(running), len(work),
                   durations, len(args.gpus), args.eta_seed_hours)
            last_report = time.time()

    report("final", completed, total, 0, 0, durations, len(args.gpus),
           args.eta_seed_hours, prefix="WALL-CLOCK")
    if failures:
        print(f"[sweep] {len(failures)} failure(s):")
        for tag, ret in failures:
            print(f"  {tag} rc={ret}")
        sys.exit(1)
    print("[sweep] all done")


if __name__ == "__main__":
    main()
