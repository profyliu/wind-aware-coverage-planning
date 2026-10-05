"""
experiments.py
==============
Full experiment matrix for the CPP benchmark study.

Runs the proposed method and two baselines over:
  - Four polygon families  × parameter sweep
  - Four cell sizes
  - Eight wind directions × four wind magnitudes

All results are logged to paper/data/results_full.csv.
Tables 2-3 (aggregated by family and wind) are exported to
paper/tables/ as CSV and LaTeX.

Usage
-----
    python experiments.py                   # full experiment matrix
    python experiments.py --pilot           # small pilot run (fast)
    python experiments.py --seeds 5         # reduce seeds per config
"""

import argparse
import csv
import itertools
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from instance_gen import generate_polygon_family
from decomposition import decompose, verify_coverage
from routing import build_route, optimal_grid_angle, optimal_grid_angle_energy, DEFAULT_POWER_MODEL

DATA_DIR = Path(__file__).parent.parent / "paper" / "data"
TABLES_DIR = Path(__file__).parent.parent / "paper" / "tables"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

POLYGON_CONFIGS = {
    "convex":       {"n_vertices_list": [6, 10, 14, 18]},
    "mild_concave": {"n_vertices_list": [10, 14, 18, 24]},
    "high_concave": {"n_notches_list":  [3, 5, 7, 8]},
    "corridor":     {"corridor_widths": [30.0, 40.0, 60.0, 80.0]},
}

CELL_SIZES_M = [10.0, 15.0, 20.0, 30.0]   # normalized to 500m diagonal polygon

WIND_DIRS_DEG = [0, 45, 90, 135, 180, 225, 270, 315]
WIND_RATIOS = [0.0, 0.1, 0.2, 0.3]         # |v_w| / v_a

AIRSPEED_MPS = 12.0
TARGET_DIAGONAL = 500.0
N_SEEDS = 20

PILOT_CELL_SIZES = [20.0]
PILOT_WIND_DIRS = [0, 90]
PILOT_WIND_RATIOS = [0.0, 0.2]
PILOT_SEEDS = 3


# ---------------------------------------------------------------------------
# Baseline 2 — non-pruned (imported from generate_paths)
# ---------------------------------------------------------------------------

def _decompose_nonpruned(polygon, cell_w, cell_h, angle_deg):
    from generate_paths import decompose_nonpruned
    return decompose_nonpruned(polygon, cell_w, cell_h, angle_deg)


# ---------------------------------------------------------------------------
# Single-instance runner
# ---------------------------------------------------------------------------

def run_instance(polygon, cell_w, cell_h, wind_deg, wind_speed, method: str,
                 objective: str = "time") -> Dict:
    """Run one (polygon, grid config, wind) experiment and return metric dict.

    Parameters
    ----------
    objective : ``'time'`` or ``'energy'``
        Controls the routing criterion for the *proposed* method:
        ``'time'``  — grid orientation and routing minimise flight time at
                      fixed nominal airspeed ``v_a``.  Energy is computed and
                      reported post-hoc using the drag-polar model.
        ``'energy'``— grid orientation and routing minimise total propulsion
                      energy; per-leg airspeed ``v_a*(θ)`` is optimised
                      simultaneously.  Flight time is also reported.
        Baselines always use time-optimal routing; energy is reported post-hoc.
    """
    airspeed = AIRSPEED_MPS
    assert airspeed > wind_speed, f"Infeasible wind ratio: {wind_speed/airspeed:.2f}"

    wind_rad = math.radians(wind_deg)
    wind_vec = np.array([wind_speed * math.cos(wind_rad),
                          wind_speed * math.sin(wind_rad)])

    t0 = time.perf_counter()
    if method == "proposed":
        if objective == "energy":
            phi_opt = optimal_grid_angle_energy(polygon, cell_w, cell_h,
                                                wind_vec, airspeed, DEFAULT_POWER_MODEL)
        else:
            phi_opt = optimal_grid_angle(polygon, cell_w, cell_h, wind_vec, airspeed)
        decomp = decompose(polygon, cell_w, cell_h, phi_opt)
    elif method == "baseline1":   # fixed 0° orientation
        phi_opt = 0.0
        decomp = decompose(polygon, cell_w, cell_h, 0.0)
    elif method == "baseline2":   # time-optimal orientation, selective pruning
        phi_opt = optimal_grid_angle(polygon, cell_w, cell_h, wind_vec, airspeed)
        decomp = decompose(polygon, cell_w, cell_h, phi_opt)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Always route with energy reporting; objective controls routing criterion
    route_obj = objective if method == "proposed" else "time"
    route = build_route(decomp, wind_vec=wind_vec, airspeed=airspeed,
                        power_model=DEFAULT_POWER_MODEL, objective=route_obj)
    runtime = time.perf_counter() - t0

    mean_va = (float(sum(route.leg_airspeeds_mps) / len(route.leg_airspeeds_mps))
               if route.leg_airspeeds_mps else airspeed)

    return {
        "method": method,
        "objective": route_obj,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "wind_deg": wind_deg,
        "wind_speed": wind_speed,
        "wind_ratio": wind_speed / airspeed,
        "phi_deg": phi_opt,
        "n_cells": decomp.n_cells,
        "n_boundary": decomp.n_boundary_cells,
        "n_interior": decomp.n_interior_cells,
        "coverage_ratio": decomp.coverage_ratio,
        "coverage_ok": verify_coverage(decomp),
        "flight_time_s": route.total_time_s,
        "path_length_m": route.total_length_m,
        "turns": route.turn_count,
        "n_strips": route.metrics["n_strips"],
        "energy_J": route.total_energy_J,
        "mean_airspeed_mps": mean_va,
        "runtime_s": runtime,
    }


# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------

def run_experiments(pilot: bool = False, n_seeds: int = N_SEEDS) -> pd.DataFrame:
    cell_sizes = PILOT_CELL_SIZES if pilot else CELL_SIZES_M
    wind_dirs = PILOT_WIND_DIRS if pilot else WIND_DIRS_DEG
    wind_ratios = PILOT_WIND_RATIOS if pilot else WIND_RATIOS
    seeds = list(range(PILOT_SEEDS if pilot else n_seeds))

    all_rows: List[Dict] = []

    # Build list of (family, param_key, param_val) triples
    tasks = []
    for family, cfg in POLYGON_CONFIGS.items():
        if "n_vertices_list" in cfg:
            for nv in cfg["n_vertices_list"]:
                tasks.append((family, "n_vertices", nv))
        elif "n_notches_list" in cfg:
            for nn in cfg["n_notches_list"]:
                tasks.append((family, "n_notches", nn))
        elif "corridor_widths" in cfg:
            for cw in cfg["corridor_widths"]:
                tasks.append((family, "corridor_width", cw))

    # methods × objectives: baselines always use time-opt; proposed uses both
    method_obj_pairs = [
        ("proposed",  "time"),
        ("proposed",  "energy"),
        ("baseline1", "time"),
        ("baseline2", "time"),
    ]
    total = len(tasks) * len(seeds) * len(cell_sizes) * len(wind_dirs) * len(wind_ratios) * len(method_obj_pairs)
    print(f"Total experiment configurations: {total}")

    with tqdm(total=total, desc="Experiments") as pbar:
        for (family, param_key, param_val), seed, cell_size, wd, wr in itertools.product(
                tasks, seeds, cell_sizes, wind_dirs, wind_ratios):

            wind_speed = AIRSPEED_MPS * wr

            # Generate polygon
            kwargs = {param_key: param_val}
            try:
                polygon = generate_polygon_family(family, seed=seed,
                                                   target_diagonal=TARGET_DIAGONAL,
                                                   **kwargs)
            except Exception as e:
                print(f"  [skip] gen failed {family}/{seed}: {e}")
                pbar.update(len(method_obj_pairs))
                continue

            base_row = {
                "family": family,
                param_key: param_val,
                "seed": seed,
                "polygon_area": polygon.area,
                "polygon_vertices": len(polygon.exterior.coords) - 1,
            }

            for method, obj in method_obj_pairs:
                try:
                    result = run_instance(polygon, cell_size, cell_size,
                                           wd, wind_speed, method, objective=obj)
                    row = {**base_row, **result}
                    all_rows.append(row)
                except Exception as e:
                    print(f"  [skip] run failed {family}/{method}/{obj}/{seed}: {e}")
                finally:
                    pbar.update(1)

    df = pd.DataFrame(all_rows)
    return df


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def make_tables(df: pd.DataFrame) -> None:
    # Table 2: Decomposition metrics by family and method/objective
    t2 = (df.groupby(["family", "method", "objective"])
            .agg(
                n_cells_mean=("n_cells", "mean"),
                n_cells_std=("n_cells", "std"),
                coverage_ratio_mean=("coverage_ratio", "mean"),
                coverage_ok_pct=("coverage_ok", lambda x: 100 * x.mean()),
                runtime_s_mean=("runtime_s", "mean"),
            )
            .reset_index()
            .round(3))
    t2_csv = TABLES_DIR / "table2_decomposition.csv"
    t2.to_csv(t2_csv, index=False)
    print(f"Saved Table 2 CSV: {t2_csv}")

    # Table 3: Route metrics by wind magnitude
    t3 = (df.groupby(["family", "method", "objective", "wind_ratio"])
            .agg(
                flight_time_s_mean=("flight_time_s", "mean"),
                flight_time_s_std=("flight_time_s", "std"),
                energy_J_mean=("energy_J", "mean"),
                energy_J_std=("energy_J", "std"),
                path_length_m_mean=("path_length_m", "mean"),
                turns_mean=("turns", "mean"),
                mean_airspeed_mps=("mean_airspeed_mps", "mean"),
            )
            .reset_index()
            .round(2))
    t3_csv = TABLES_DIR / "table3_route_metrics.csv"
    t3.to_csv(t3_csv, index=False)
    print(f"Saved Table 3 CSV: {t3_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run CPP benchmark experiments")
    parser.add_argument("--pilot", action="store_true",
                         help="Run small pilot study (fast)")
    parser.add_argument("--seeds", type=int, default=N_SEEDS,
                         help=f"Seeds per config (default {N_SEEDS})")
    parser.add_argument("--no-tables", action="store_true",
                         help="Skip table generation")
    args = parser.parse_args()

    tag = "pilot" if args.pilot else "full"
    df = run_experiments(pilot=args.pilot, n_seeds=args.seeds)

    csv_path = DATA_DIR / f"results_{tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results: {csv_path}  ({len(df)} rows)")

    if not args.no_tables:
        make_tables(df)


if __name__ == "__main__":
    main()



# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------

def run_experiments(pilot: bool = False, n_seeds: int = N_SEEDS) -> pd.DataFrame:
    cell_sizes = PILOT_CELL_SIZES if pilot else CELL_SIZES_M
    wind_dirs = PILOT_WIND_DIRS if pilot else WIND_DIRS_DEG
    wind_ratios = PILOT_WIND_RATIOS if pilot else WIND_RATIOS
    seeds = list(range(PILOT_SEEDS if pilot else n_seeds))

    all_rows: List[Dict] = []

    # Build list of (family, param_key, param_val) triples
    tasks = []
    for family, cfg in POLYGON_CONFIGS.items():
        if "n_vertices_list" in cfg:
            for nv in cfg["n_vertices_list"]:
                tasks.append((family, "n_vertices", nv))
        elif "n_notches_list" in cfg:
            for nn in cfg["n_notches_list"]:
                tasks.append((family, "n_notches", nn))
        elif "corridor_widths" in cfg:
            for cw in cfg["corridor_widths"]:
                tasks.append((family, "corridor_width", cw))

    total = len(tasks) * len(seeds) * len(cell_sizes) * len(wind_dirs) * len(wind_ratios) * 3
    print(f"Total experiment configurations: {total}")

    with tqdm(total=total, desc="Experiments") as pbar:
        for (family, param_key, param_val), seed, cell_size, wd, wr in itertools.product(
                tasks, seeds, cell_sizes, wind_dirs, wind_ratios):

            wind_speed = AIRSPEED_MPS * wr

            # Generate polygon
            kwargs = {param_key: param_val}
            try:
                polygon = generate_polygon_family(family, seed=seed,
                                                   target_diagonal=TARGET_DIAGONAL,
                                                   **kwargs)
            except Exception as e:
                print(f"  [skip] gen failed {family}/{seed}: {e}")
                pbar.update(3)
                continue

            base_row = {
                "family": family,
                param_key: param_val,
                "seed": seed,
                "polygon_area": polygon.area,
                "polygon_vertices": len(polygon.exterior.coords) - 1,
            }

            for method in ["proposed", "baseline1", "baseline2"]:
                try:
                    result = run_instance(polygon, cell_size, cell_size,
                                           wd, wind_speed, method)
                    row = {**base_row, **result}
                    all_rows.append(row)
                except Exception as e:
                    print(f"  [skip] run failed {family}/{method}/{seed}: {e}")
                finally:
                    pbar.update(1)

    df = pd.DataFrame(all_rows)
    return df


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def make_tables(df: pd.DataFrame) -> None:
    # Table 2: Decomposition metrics by family
    t2 = (df.groupby(["family", "method"])
            .agg(
                n_cells_mean=("n_cells", "mean"),
                n_cells_std=("n_cells", "std"),
                coverage_ratio_mean=("coverage_ratio", "mean"),
                coverage_ok_pct=("coverage_ok", lambda x: 100 * x.mean()),
                runtime_s_mean=("runtime_s", "mean"),
            )
            .reset_index()
            .round(3))
    t2_csv = TABLES_DIR / "table2_decomposition.csv"
    t2.to_csv(t2_csv, index=False)
    print(f"Saved Table 2 CSV: {t2_csv}")

    # LaTeX version
    t2_tex = TABLES_DIR / "table2_decomposition.tex"
    with open(t2_tex, "w") as f:
        f.write("% Table 2 – Decomposition metrics (auto-generated)\n")
        f.write(t2.to_latex(index=False, float_format="%.3f",
                              caption="Decomposition metrics by polygon family and method.",
                              label="tab:decomposition"))
    print(f"Saved Table 2 LaTeX: {t2_tex}")

    # Table 3: Route metrics by wind magnitude
    t3 = (df.groupby(["family", "method", "wind_ratio"])
            .agg(
                flight_time_s_mean=("flight_time_s", "mean"),
                flight_time_s_std=("flight_time_s", "std"),
                path_length_m_mean=("path_length_m", "mean"),
                turns_mean=("turns", "mean"),
            )
            .reset_index()
            .round(2))
    t3_csv = TABLES_DIR / "table3_route_metrics.csv"
    t3.to_csv(t3_csv, index=False)
    print(f"Saved Table 3 CSV: {t3_csv}")

    t3_tex = TABLES_DIR / "table3_route_metrics.tex"
    with open(t3_tex, "w") as f:
        f.write("% Table 3 – Route metrics (auto-generated)\n")
        f.write(t3.to_latex(index=False, float_format="%.2f",
                              caption="Route metrics by polygon family, method, and wind magnitude.",
                              label="tab:route"))
    print(f"Saved Table 3 LaTeX: {t3_tex}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run CPP benchmark experiments")
    parser.add_argument("--pilot", action="store_true",
                         help="Run small pilot study (fast)")
    parser.add_argument("--seeds", type=int, default=N_SEEDS,
                         help=f"Seeds per config (default {N_SEEDS})")
    parser.add_argument("--no-tables", action="store_true",
                         help="Skip table generation")
    args = parser.parse_args()

    tag = "pilot" if args.pilot else "full"
    df = run_experiments(pilot=args.pilot, n_seeds=args.seeds)

    csv_path = DATA_DIR / f"results_{tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results: {csv_path}  ({len(df)} rows)")

    if not args.no_tables:
        make_tables(df)


if __name__ == "__main__":
    main()
