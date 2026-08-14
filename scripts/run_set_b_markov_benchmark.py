#!/usr/bin/env python3
"""Run benchmark of Markov + LAHC native solver across Set B instances."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time

INSTANCES = [
    "V2.12.xml",
    "V2.13.xml",
    "V2.14.xml",
    "V2.25.xml",
]

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=int, default=180)
    parser.add_argument("--instances-dir", type=Path, default=Path("roadef_2016_data/set_B/Instances_B_V25-11042016"))
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/markov_benchmark_results"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Starting Set B Benchmark ({len(INSTANCES)} instances, {args.time_limit}s limit each) ===", flush=True)

    summary = []
    for inst_name in INSTANCES:
        inst_path = args.instances_dir / inst_name
        out_path = args.output_dir / inst_name
        print(f"\n--- Solving {inst_name} ({args.time_limit}s budget) ---", flush=True)
        
        cmd = [
            "uv", "run", "vrp-solver", "native-solve",
            str(inst_path), str(out_path),
            "--time-limit", str(args.time_limit),
            "--use-lahc", "--use-markov",
            "--iterations", "96",
            "--candidates-per-move", "100",
            "--workers", "2",
            "--restart-rounds", "2",
        ]
        t0 = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.monotonic() - t0
        print(f"Finished {inst_name} in {elapsed:.1f}s. Exit code: {proc.returncode}")

        # Check official checker validity
        verify_cmd = ["uv", "run", "vrp-solver", "verify-official", str(inst_path), str(out_path)]
        v_proc = subprocess.run(verify_cmd, capture_output=True, text=True)
        is_valid = "official_valid,True" in v_proc.stdout
        lr = "N/A"
        for line in v_proc.stdout.splitlines():
            if line.startswith("official_logistic_ratio,"):
                lr = line.split(",")[-1].strip()

        print(f"Official Result {inst_name}: Valid={is_valid}, LR={lr}", flush=True)
        summary.append((inst_name, elapsed, is_valid, lr))

    print("\n" + "=" * 60)
    print("=== BENCHMARK SUMMARY ===")
    print("=" * 60)
    for name, el, val, lr in summary:
        status_str = "VALID" if val else "INVALID"
        print(f"{name:15s} | Time: {el:6.1f}s | Status: {status_str:7s} | LR: {lr}")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
