"""AirfRANS dataset loader using the LIPS benchmark framework.

Wraps the LIPS `AirfRANSBenchmark` so paths obey the project layout:

    /workspace/space_pinn/data/airfrans/Dataset/         # raw simulations
    /workspace/space_pinn/data/airfrans/lips_logs.log    # LIPS logging
    /workspace/space_pinn/src/geompnn/airfrans_benchmark.ini   # 2024 ML4CFD bench config
"""

from __future__ import annotations

import os
import pathlib

PKG_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent.parent  # /workspace/space_pinn

DATA_ROOT = PROJECT_ROOT / "data" / "airfrans"
DATASET_DIR = DATA_ROOT / "Dataset"
LOG_PATH = DATA_ROOT / "lips_logs.log"
BENCH_CONFIG_PATH = PKG_DIR / "airfrans_benchmark.ini"
BENCHMARK_NAME = "Case1"


def download_if_needed():
    """Download AirfRANS dataset to DATASET_DIR if not already present.

    AirfRANS is large (~17 GB compressed), this can take a long time.
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        return DATASET_DIR
    from lips.dataset.airfransDataSet import download_data
    download_data(root_path=str(DATA_ROOT), directory_name="Dataset")
    return DATASET_DIR


def get_benchmark():
    """Construct and load AirfRANSBenchmark on the 2024 ML4CFD splits.

    Returns the loaded benchmark; train_dataset has 103 simulations,
    _test_dataset has 200 (in-distribution), _test_ood_dataset has 496.
    """
    from lips.benchmark.airfransBenchmark import AirfRANSBenchmark

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not BENCH_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing benchmark config at {BENCH_CONFIG_PATH}. "
            "It should ship with the repo (copied from ML4CFD starting kit)."
        )

    benchmark = AirfRANSBenchmark(
        benchmark_path=str(DATASET_DIR),
        config_path=str(BENCH_CONFIG_PATH),
        benchmark_name=BENCHMARK_NAME,
        log_path=str(LOG_PATH),
    )
    benchmark.load(path=str(DATASET_DIR))
    return benchmark


def benchmark_split_sizes(benchmark) -> dict:
    """Return number of simulations in each split. Useful for sanity checks."""
    return {
        "train": len(benchmark.train_dataset.get_simulations_sizes()),
        "test": len(benchmark._test_dataset.get_simulations_sizes()),
        "test_ood": len(benchmark._test_ood_dataset.get_simulations_sizes()),
    }
