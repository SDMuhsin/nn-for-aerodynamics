"""Train one ablation × one seed of GeoMPNN.

Usage:
    python -m geompnn.train \
        --config src/geompnn/configs/s2v.json \
        --seed 0 \
        --output_dir results/s2v/seed0 \
        [--override nb_epochs=5 subsampling=2000]   # for smoke runs

Side effects under output_dir:
    parameters.json   # frozen config used for the run
    ckpt.pt           # final model state_dict
    scaler files      # per-field LIPS scaler dumps (from simulator code)
    eval.json         # raw fc_metrics_test, fc_metrics_test_ood, timings
    score.json        # computed ML/Physics/OOD/Global from scoring.py
    train.log         # tee'd stdout
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from geompnn import data as data_mod
from geompnn import scoring


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_overrides(items):
    """`['nb_epochs=5', 'subsampling=2000']` -> dict with auto-typed values."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--override entry needs key=value, got {item!r}")
        k, v = item.split("=", 1)
        # auto-type: bool, int, float, json, else string
        low = v.lower()
        if low in {"true", "false"}:
            out[k] = low == "true"
            continue
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = json.loads(v)
            continue
        except (ValueError, json.JSONDecodeError):
            pass
        out[k] = v
    return out


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to ablation parameters.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--override",
        nargs="+",
        default=None,
        help="Inline overrides for simulator_extra_parameters (e.g. nb_epochs=5).",
    )
    parser.add_argument(
        "--limit_train_simulations",
        type=int,
        default=None,
        help="If set, truncate train_dataset to first N simulations (smoke runs only).",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip benchmark evaluation (smoke debug).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text())
    overrides = parse_overrides(args.override)
    cfg["simulator_extra_parameters"].update(overrides)
    cfg["_seed"] = args.seed
    cfg["_overrides"] = overrides
    write_json(out_dir / "parameters.json", cfg)

    log_path = out_dir / "train.log"
    log_file = open(log_path, "w")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, x):
            for s in self.streams:
                s.write(x)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"[geompnn.train] config={args.config} seed={args.seed} out={out_dir}")
    print(f"[geompnn.train] overrides={overrides}")
    set_seed(args.seed)

    benchmark = data_mod.get_benchmark()
    sizes = data_mod.benchmark_split_sizes(benchmark)
    print(f"[geompnn.train] split sizes: {sizes}")
    expected = {"train": 103, "test": 200, "test_ood": 496}
    if sizes != expected:
        print(f"[geompnn.train] WARNING: split sizes {sizes} != expected {expected}")

    if args.limit_train_simulations is not None:
        from lips.dataset.airfransDataSet import extract_dataset_by_simulations

        n = args.limit_train_simulations
        sim_indices = list(range(n))
        benchmark.train_dataset = extract_dataset_by_simulations(
            newdataset_name=benchmark.train_dataset.name,
            dataset=benchmark.train_dataset,
            simulation_indices=sim_indices,
        )
        print(f"[geompnn.train] truncated train to first {n} simulations (smoke mode)")

    # Lazy import so simulator's torch_geometric / LIPS imports happen after data load
    from geompnn.simulator import AugmentedSimulator

    extra = cfg["simulator_extra_parameters"]
    simulator = AugmentedSimulator(benchmark=benchmark, **extra)

    # Hook ckpt path into simulator (it normally derives from a TensorBoard writer)
    simulator.ckpt_path = str(out_dir)
    if simulator.logscale_press:
        simulator.pressure_scaler_path = str(out_dir / "pressure_scaler")
        simulator.log_pressure_scaler_path = str(out_dir / "log_pressure_scaler")

    train_start = time.time()
    simulator.train(benchmark.train_dataset)
    training_time = time.time() - train_start
    print(f"[geompnn.train] training_time={training_time:.1f}s")

    if args.skip_eval:
        print("[geompnn.train] --skip_eval set; exiting after training")
        return

    print("[geompnn.train] evaluating on test (in-distribution)")
    t0 = time.time()
    fc_metrics_test = benchmark.evaluate_simulator(
        dataset="test", augmented_simulator=simulator, eval_batch_size=256000
    )
    test_eval_time = time.time() - t0
    test_mean_sim_time = test_eval_time / len(benchmark._test_dataset.get_simulations_sizes())

    print("[geompnn.train] evaluating on test_ood")
    t0 = time.time()
    fc_metrics_test_ood = benchmark.evaluate_simulator(
        dataset="test_ood", augmented_simulator=simulator, eval_batch_size=256000
    )
    test_ood_eval_time = time.time() - t0
    test_ood_mean_sim_time = test_ood_eval_time / len(
        benchmark._test_ood_dataset.get_simulations_sizes()
    )

    eval_blob = {
        "training_time": training_time,
        "test_evaluation_time": test_eval_time,
        "test_mean_simulation_time": test_mean_sim_time,
        "test_ood_evaluation_time": test_ood_eval_time,
        "test_ood_mean_simulation_time": test_ood_mean_sim_time,
        "fc_metrics_test": fc_metrics_test,
        "fc_metrics_test_ood": fc_metrics_test_ood,
    }
    write_json(out_dir / "eval.json", eval_blob)

    score = scoring.score_from_lips_output(
        fc_metrics_test=fc_metrics_test,
        fc_metrics_test_ood=fc_metrics_test_ood,
        test_mean_simulation_time=test_mean_sim_time,
        test_ood_mean_simulation_time=test_ood_mean_sim_time,
    )
    write_json(out_dir / "score.json", score)
    print(f"[geompnn.train] score: ML={score['ML']:.2f} Physics={score['Physics']:.2f} "
          f"OOD={score['OOD']:.2f} Global={score['Global']:.2f}")


if __name__ == "__main__":
    main()
