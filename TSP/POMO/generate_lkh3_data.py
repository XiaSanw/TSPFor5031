"""
Generate random TSP instances, solve with LKH3, save as (problem, tour) pairs.
Supports multi-process parallel solving.

Usage:
    cd TSP/POMO
    export LKH3_PATH="$HOME/LKH-3.0.10/LKH"
    python generate_lkh3_data.py
"""
import os
import sys
import subprocess
import torch
import numpy as np
from multiprocessing import Pool

# Ensure parent dir is on path for TSProblemDef import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

LKH3_BIN = os.environ.get("LKH3_PATH", "./LKH")
DATA_DIR = "./lkh3_expert_data"
NUM_WORKERS = 8


def problem_to_tsplib(problem, filename):
    """problem: (N, 2) numpy array, coordinates in [0, 1] -> TSPLIB .tsp file"""
    n = problem.shape[0]
    scale = 1000.0
    with open(filename, 'w') as f:
        f.write(f"NAME: random_{n}\n")
        f.write(f"TYPE: TSP\n")
        f.write(f"DIMENSION: {n}\n")
        f.write(f"EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write(f"NODE_COORD_SECTION\n")
        for i in range(n):
            f.write(f"{i+1} {problem[i, 0]*scale:.6f} {problem[i, 1]*scale:.6f}\n")
        f.write("EOF\n")


def run_lkh3(tsp_file, tour_file, par_file):
    """Call LKH3 to solve a .tsp file"""
    with open(par_file, 'w') as f:
        f.write(f"PROBLEM_FILE = {tsp_file}\n")
        f.write(f"TOUR_FILE = {tour_file}\n")
        f.write(f"RUNS = 1\n")
        f.write(f"SEED = 0\n")

    result = subprocess.run(
        [LKH3_BIN, par_file],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"LKH3 failed: {result.stderr}")


def parse_tour_file(tour_file, n):
    """Parse LKH3 .tour file, return 0-based city index list"""
    tour = []
    in_tour = False
    with open(tour_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "TOUR_SECTION":
                in_tour = True
                continue
            if line == "-1" or line == "EOF":
                break
            if in_tour and line:
                tour.append(int(line) - 1)  # 1-based -> 0-based
    assert len(tour) == n, f"Tour length mismatch: {len(tour)} != {n}"
    return tour


def solve_single_instance(args):
    """Single instance solving pipeline (called by multiprocessing workers)"""
    i, problem, problem_size, output_dir = args
    tsp_file = os.path.join(output_dir, f"prob_{i}.tsp")
    tour_file = os.path.join(output_dir, f"prob_{i}.tour")
    par_file = os.path.join(output_dir, f"prob_{i}.par")

    problem_to_tsplib(problem, tsp_file)
    run_lkh3(tsp_file, tour_file, par_file)
    tour = parse_tour_file(tour_file, problem_size)

    # Clean up temp files
    os.remove(tsp_file)
    os.remove(tour_file)
    os.remove(par_file)

    return {'problem': problem, 'tour': np.array(tour, dtype=np.int64)}


def generate_dataset(num_instances, problem_size, output_dir, num_workers=NUM_WORKERS):
    """Parallel generate a batch of instances and solve with LKH3"""
    os.makedirs(output_dir, exist_ok=True)
    from TSProblemDef import get_random_problems

    # .cpu() needed: get_random_problems may return CUDA tensor
    problems = get_random_problems(num_instances, problem_size).cpu().numpy()
    # shape: (num_instances, problem_size, 2)

    args_list = [
        (i, problems[i], problem_size, output_dir)
        for i in range(num_instances)
    ]

    all_data = []
    with Pool(num_workers) as pool:
        for result in pool.imap_unordered(solve_single_instance, args_list):
            all_data.append(result)
            if len(all_data) % 1000 == 0:
                print(f"  Progress: {len(all_data)}/{num_instances}")

    save_path = os.path.join(output_dir, f"lkh3_data_n{problem_size}.pt")
    torch.save(all_data, save_path)
    print(f"Saved {num_instances} instances to {save_path}")


if __name__ == "__main__":
    # Adjust data sizes based on your time/compute budget
    # n=100 fastest (~2h/50k instances 8 cores), n=200 slower (~6h/50k instances 8 cores)
    generate_dataset(num_instances=50000, problem_size=100,
                     output_dir=f"{DATA_DIR}/n100")
    generate_dataset(num_instances=50000, problem_size=200,
                     output_dir=f"{DATA_DIR}/n200")
    generate_dataset(num_instances=30000, problem_size=150,
                     output_dir=f"{DATA_DIR}/n150")
