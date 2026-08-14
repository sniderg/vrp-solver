"""Run parallel cluster construction and multi-week rolling rescue for remaining Set B instances."""
import subprocess
import time
from pathlib import Path

INSTANCES = [
    ("V2.19", "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.19.xml", 35),
    ("V2.18", "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.18.xml", 35),
    ("V2.20.2", "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.20.2.xml", 35),
    ("V2.21.2", "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.21.2.xml", 35),
    ("V2.22", "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.22.xml", 21),
    ("V2.23", "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.23.xml", 21),
]

def run_instance(name: str, path: str, horizon_days: int) -> None:
    cluster_xml = Path(f"scratch/{name.lower()}_initial_cluster.xml")
    rolling_xml = Path(f"scratch/{name.lower()}_rolling_result.xml")
    log_file = Path(f"scratch/{name.lower()}_solve.log")

    print(f"[{name}] Starting cluster construction...")
    subprocess.run(
        [
            "uv", "run", "vrp-solver", "cluster-construct-solution",
            path, str(cluster_xml)
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(f"[{name}] Cluster construction complete. Starting rolling rescue ({horizon_days}d)...")
    
    with open(log_file, "w") as out:
        try:
            subprocess.run(
                [
                    "uv", "run", "vrp-solver", "robust-rolling-rescue",
                    path, str(cluster_xml), str(rolling_xml),
                    "--mode", "deterministic",
                    "--horizon-days", str(horizon_days),
                    "--commit-days", "7",
                    "--lookahead-days", "3",
                    "--cg-iterations", "3",
                    "--selector-time-limit", "60",
                    "--multi-reload-columns",
                ],
                check=True,
                stdout=out,
                stderr=out,
                timeout=1800,  # Strict 30-minute solve cap
            )
            print(f"[{name}] Rolling rescue finished -> {rolling_xml}")
        except subprocess.TimeoutExpired:
            print(f"[{name}] Reached 30-minute time cap; proceeding with best intermediate solution.")
        except Exception as e:
            print(f"[{name}] Error during solve: {e}")


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    print(f"Launching parallel solves for {len(INSTANCES)} remaining Set B instances...")
    # Use max 4 parallel processes to leave headroom on 8-core CPU
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(run_instance, name, path, horizon)
            for name, path, horizon in INSTANCES
        ]
        for f in futures:
            try:
                f.result()
            except Exception as e:
                print(f"Error in solve: {e}")
    print("All parallel solves completed.")
