"""Re-evaluate a saved checkpoint without re-training.

Two modes:

  1. Standard re-evaluation (full mesh, 2024 ML4CFD scoring):
       python -m geompnn.eval --run_dir results/s2v/seed0

  2. Distribution-shift study (Fig 3/5):
       python -m geompnn.eval --run_dir results/s2v/seed0 \\
           --shift_eval --subsampling 32000

       Re-runs both `test` and `test_ood` evaluation while subsampling each
       full-mesh simulation to N volume nodes (matching training conditions).
       Writes shift_eval.json with raw metrics under both regimes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from geompnn import data as data_mod
from geompnn import scoring


def load_simulator(run_dir: Path, benchmark, override_subsampling: int | None = None):
    cfg = json.loads((run_dir / "parameters.json").read_text())
    extra = dict(cfg["simulator_extra_parameters"])
    if override_subsampling is not None:
        extra["subsampling"] = override_subsampling

    from geompnn.simulator import AugmentedSimulator

    simulator = AugmentedSimulator(benchmark=benchmark, **extra)
    simulator.ckpt_path = str(run_dir)
    if simulator.logscale_press:
        simulator.pressure_scaler_path = str(run_dir / "pressure_scaler")
        simulator.log_pressure_scaler_path = str(run_dir / "log_pressure_scaler")
    simulator.restore(str(run_dir))
    return simulator, cfg


def evaluate_full(simulator, benchmark) -> dict:
    print("[geompnn.eval] full-mesh evaluation")
    t0 = time.time()
    test_metrics = benchmark.evaluate_simulator(
        dataset="test", augmented_simulator=simulator, eval_batch_size=256000
    )
    test_time = time.time() - t0
    test_mean = test_time / len(benchmark._test_dataset.get_simulations_sizes())

    t0 = time.time()
    test_ood_metrics = benchmark.evaluate_simulator(
        dataset="test_ood", augmented_simulator=simulator, eval_batch_size=256000
    )
    test_ood_time = time.time() - t0
    test_ood_mean = test_ood_time / len(benchmark._test_ood_dataset.get_simulations_sizes())

    return {
        "test_evaluation_time": test_time,
        "test_mean_simulation_time": test_mean,
        "test_ood_evaluation_time": test_ood_time,
        "test_ood_mean_simulation_time": test_ood_mean,
        "fc_metrics_test": test_metrics,
        "fc_metrics_test_ood": test_ood_metrics,
    }


def write_score(out_dir: Path, eval_blob: dict, name: str = "score.json"):
    score = scoring.score_from_lips_output(
        fc_metrics_test=eval_blob["fc_metrics_test"],
        fc_metrics_test_ood=eval_blob["fc_metrics_test_ood"],
        test_mean_simulation_time=eval_blob["test_mean_simulation_time"],
        test_ood_mean_simulation_time=eval_blob["test_ood_mean_simulation_time"],
    )
    (out_dir / name).write_text(json.dumps(score, indent=2, default=str))
    print(f"[geompnn.eval] wrote {out_dir / name}: Global={score['Global']:.2f}")
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument(
        "--shift_eval",
        action="store_true",
        help="Compare full-mesh vs subsampled (Fig 3/5).",
    )
    parser.add_argument(
        "--subsampling",
        type=int,
        default=None,
        help="If set, override config subsampling for this evaluation.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    benchmark = data_mod.get_benchmark()

    if args.shift_eval:
        # First: full-mesh (subsampling=full)
        simulator, _ = load_simulator(run_dir, benchmark, override_subsampling=None)
        # In simulator.predict the loader is built with subsample=False (full mesh).
        # benchmark.evaluate_simulator calls simulator.predict, so this is full mesh.
        full_blob = evaluate_full(simulator, benchmark)
        # Second: 32K-subsampled — we have to use a manual evaluation loop because
        # benchmark.evaluate_simulator triggers full-mesh prediction. We'll add this
        # path once distribution-shift step is reached.
        out = {"full_mesh": full_blob}
        (run_dir / "shift_eval_full.json").write_text(json.dumps(full_blob, indent=2, default=str))
        print("[geompnn.eval] shift_eval mode: full-mesh saved; subsampled evaluation TODO")
        write_score(run_dir, full_blob, name="shift_score_full.json")
        return

    simulator, _ = load_simulator(run_dir, benchmark, override_subsampling=args.subsampling)
    eval_blob = evaluate_full(simulator, benchmark)
    (run_dir / "eval.json").write_text(json.dumps(eval_blob, indent=2, default=str))
    write_score(run_dir, eval_blob)


if __name__ == "__main__":
    main()
