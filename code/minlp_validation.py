"""
minlp_validation.py
===================
Exact solver (Held-Karp DP) for the strip-ordering subproblem and comparison
against the heuristic routing to validate the MINLP formulation.

For a fixed grid orientation φ, the inner decision variables are:
  - strip traversal directions  d_i ∈ {0, 1}  (0 = left-to-right, 1 = right-to-left)
  - strip ordering              σ              (permutation of strips)
  - per-leg airspeeds           v_{a,ell}      (continuous, optimised independently)

Key structural observation
--------------------------
For fixed φ and fixed strip d_i, the energy-optimal airspeed can be chosen
independently for each strip or transition leg.  This decouples speed optimisation
from ordering/direction, so the residual problem is a small
Asymmetric Travelling-Salesman-Path (ATSP-P) over r strips, each with two direction
choices.  The ATSP-P is solved exactly by the Held-Karp algorithm
(Held & Karp, 1962) with states (last_strip, last_direction, visited_set), running
in O(r² × 4 × 2^r) time.

Pipeline
--------
1. Generate several small polygon instances (n_strips ≈ 6–12).
2. For each instance and orientation φ* (taken from the heuristic):
   a. Build the strips and pre-compute E_i^strip(d) and E_{ij}^trans(d_i, d_j).
   b. Run the heuristic (build_route with energy objective).
   c. Run the Held-Karp exact solver.
3. Compare heuristic vs. optimal energy; report gaps and timing.
4. Write summary CSV to paper/data/minlp_validation.csv.

Usage
-----
    python3 minlp_validation.py
    python3 minlp_validation.py --verbose
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from decomposition import decompose, Cell, DecompositionResult
from routing import (
    build_route, optimal_grid_angle_energy, energy_cost,
    DEFAULT_POWER_MODEL, PowerModel, Strip,
)
from instance_gen import generate_polygon_family

DATA_DIR = Path(__file__).parent.parent / "paper" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pre-computation helpers
# ---------------------------------------------------------------------------

def _strip_energy_optimal(strip: Strip, d: int, wind_vec: np.ndarray,
                           power_model: PowerModel) -> float:
    """
    Energy for strip traversal in direction d at per-leg optimal airspeeds.

    d=0: left-to-right (index 0 → last); d=1: right-to-left (last → 0).
    """
    cells = strip.cells if d == 0 else list(reversed(strip.cells))
    total = 0.0
    for k in range(len(cells) - 1):
        total += energy_cost(cells[k].center, cells[k + 1].center,
                             wind_vec, power_model)
    return total


def _transition_energy(s_from: Strip, d_from: int,
                        s_to: Strip, d_to: int,
                        wind_vec: np.ndarray,
                        power_model: PowerModel) -> float:
    """
    Energy for the transition from s_from (leaving in direction d_from) to
    s_to (arriving from direction d_to) at the energy-optimal airspeed.

    The exit point of s_from under direction d_from:
      d_from=0 (L-to-R): exit = rightmost cell centre
      d_from=1 (R-to-L): exit = leftmost cell centre
    The entry point of s_to under direction d_to:
      d_to=0 (L-to-R): entry = leftmost cell centre
      d_to=1 (R-to-L): entry = rightmost cell centre
    """
    exit_from = (s_from.cells[-1].center if d_from == 0
                 else s_from.cells[0].center)
    entry_to  = (s_to.cells[0].center   if d_to == 0
                 else s_to.cells[-1].center)
    return energy_cost(exit_from, entry_to, wind_vec, power_model)


def precompute_costs(
        strips: List[Strip],
        wind_vec: np.ndarray,
        power_model: PowerModel,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute strip-energy and transition-energy tables.

    Returns
    -------
    strip_E : (r, 2) array  — strip_E[i, d] = strip energy for strip i, direction d
    trans_E : (r, r, 2, 2) array — trans_E[i, j, di, dj] = transition energy
    """
    r = len(strips)
    strip_E = np.zeros((r, 2))
    trans_E = np.zeros((r, r, 2, 2))

    for i, s in enumerate(strips):
        for d in range(2):
            strip_E[i, d] = _strip_energy_optimal(s, d, wind_vec, power_model)

    for i in range(r):
        for j in range(r):
            if i == j:
                continue
            for di in range(2):
                for dj in range(2):
                    trans_E[i, j, di, dj] = _transition_energy(
                        strips[i], di, strips[j], dj,
                        wind_vec, power_model,
                    )

    return strip_E, trans_E


# ---------------------------------------------------------------------------
# Held-Karp exact solver
# ---------------------------------------------------------------------------

def held_karp_exact(strip_E: np.ndarray, trans_E: np.ndarray) -> Tuple[float, List[int], List[int]]:
    """
    Exact Held-Karp dynamic programming for the Hamiltonian path with direction choice.

    State: (last_strip i, last_direction di, visited_set S as bitmask)
    dp[S][i][di] = minimum energy to have visited exactly the strips in S,
                   finishing at strip i in direction di.
    Initial: dp[{i}][i][di] = strip_E[i, di]  for each i, di
    Transition: dp[S | (1<<j)][j][dj]
                = min_{i in S, di} ( dp[S][i][di] + trans_E[i,j,di,dj] ) + strip_E[j, dj]
    Answer: min over all final states (S_full)[i][di]

    Complexity: O(r² × 4 × 2^r)  — feasible for r ≤ 15 in <1 s.

    Parameters
    ----------
    strip_E  : (r, 2)       pre-computed strip energies
    trans_E  : (r, r, 2, 2) pre-computed transition energies

    Returns
    -------
    opt_energy   : float — optimal total energy (J)
    opt_order    : list of strip indices in visit order
    opt_dirs     : list of directions (0/1) for each strip
    """
    r = strip_E.shape[0]
    full = (1 << r) - 1

    INF = float('inf')
    # dp[S][i][di] and parent[S][i][di] = (prev_i, prev_di)
    dp = [[[INF, INF] for _ in range(r)] for _ in range(1 << r)]
    parent = [[[(None, None), (None, None)] for _ in range(r)] for _ in range(1 << r)]

    # Initialise single-strip states
    for i in range(r):
        for di in range(2):
            dp[1 << i][i][di] = float(strip_E[i, di])

    # Fill DP
    for S in range(1, 1 << r):
        for i in range(r):
            if not (S >> i & 1):
                continue
            for di in range(2):
                if dp[S][i][di] == INF:
                    continue
                cur = dp[S][i][di]
                # Extend to any unvisited strip j
                for j in range(r):
                    if S >> j & 1:
                        continue
                    S2 = S | (1 << j)
                    for dj in range(2):
                        new_cost = cur + float(trans_E[i, j, di, dj]) + float(strip_E[j, dj])
                        if new_cost < dp[S2][j][dj]:
                            dp[S2][j][dj] = new_cost
                            parent[S2][j][dj] = (i, di)

    # Find best final state
    opt_energy = INF
    opt_last_i = 0
    opt_last_di = 0
    for i in range(r):
        for di in range(2):
            if dp[full][i][di] < opt_energy:
                opt_energy = dp[full][i][di]
                opt_last_i = i
                opt_last_di = di

    # Reconstruct path
    opt_order = []
    opt_dirs  = []
    S = full
    i, di = opt_last_i, opt_last_di
    while S:
        opt_order.append(i)
        opt_dirs.append(di)
        prev = parent[S][i][di]
        S = S ^ (1 << i)
        if prev[0] is None:
            break
        i, di = prev

    opt_order.reverse()
    opt_dirs.reverse()
    return opt_energy, opt_order, opt_dirs


# ---------------------------------------------------------------------------
# Instance runner
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    polygon_family: str
    seed: int
    n_cells: int
    n_strips: int
    phi_deg: float
    heuristic_energy_J: float
    optimal_energy_J: float
    gap_pct: float
    heuristic_time_s: float
    exact_time_s: float
    wind_speed_mps: float
    wind_dir_deg: float


def run_validation(polygon_family: str, seed: int,
                   cell_w: float, cell_h: float,
                   wind_dir_deg: float, wind_speed: float,
                   power_model: PowerModel,
                   verbose: bool = False) -> Optional[ValidationResult]:
    """Run one instance: heuristic vs. exact (Held-Karp) comparison."""
    wind_rad = math.radians(wind_dir_deg)
    wind_vec = np.array([wind_speed * math.cos(wind_rad),
                         wind_speed * math.sin(wind_rad)])
    airspeed = max(14.0, wind_speed * 1.1 + 8.0)  # feasibility margin

    try:
        polygon = generate_polygon_family(polygon_family, seed=seed)
    except Exception as e:
        if verbose:
            print(f"  [skip] cannot generate {polygon_family}/seed={seed}: {e}")
        return None

    # --- Heuristic: find energy-optimal orientation and route ---
    t_h0 = time.perf_counter()
    phi_star = optimal_grid_angle_energy(polygon, cell_w, cell_h,
                                         wind_vec, airspeed, power_model,
                                         n_angles=18)
    decomp = decompose(polygon, cell_w, cell_h, phi_star)
    if len(decomp.cells) == 0:
        return None

    heuristic_route = build_route(decomp, wind_vec=wind_vec, airspeed=airspeed,
                                   power_model=power_model, objective='energy')
    t_h1 = time.perf_counter()
    heuristic_time = t_h1 - t_h0
    heuristic_energy = heuristic_route.total_energy_J

    strips = heuristic_route.strips
    r = len(strips)
    if r < 2:
        if verbose:
            print(f"  [skip] only {r} strip(s), skipping exact solve")
        return None

    n_cells = len(decomp.cells)

    # --- Pre-compute energy tables ---
    strip_E, trans_E = precompute_costs(strips, wind_vec, power_model)

    # Heuristic total energy recomputed from tables (for apples-to-apples comparison)
    # Note: heuristic_route uses optimal airspeed for transitions; we re-evaluate
    # the heuristic ordering using the same transition model as the exact solver.
    heuristic_order  = list(range(r))   # strips are already in route order
    heuristic_dirs   = [0 if s.direction == 1 else 1 for s in strips]

    heuristic_table_energy = sum(strip_E[heuristic_order[k],
                                          heuristic_dirs[k]]
                                  for k in range(r))
    for k in range(r - 1):
        i = heuristic_order[k];  di = heuristic_dirs[k]
        j = heuristic_order[k + 1]; dj = heuristic_dirs[k + 1]
        heuristic_table_energy += trans_E[i, j, di, dj]

    # --- Exact solver ---
    t_e0 = time.perf_counter()
    opt_energy, opt_order, opt_dirs = held_karp_exact(strip_E, trans_E)
    t_e1 = time.perf_counter()
    exact_time = t_e1 - t_e0

    gap_pct = 100.0 * (heuristic_table_energy - opt_energy) / max(opt_energy, 1.0)

    if verbose:
        print(f"  {polygon_family}/seed={seed}  r={r}  cells={n_cells}  "
              f"φ*={phi_star:.0f}°  wind={wind_speed:.1f} m/s @ {wind_dir_deg:.0f}°")
        print(f"    heuristic: {heuristic_table_energy/1e3:.3f} kJ  "
              f"exact: {opt_energy/1e3:.3f} kJ  "
              f"gap: {gap_pct:.2f}%  "
              f"exact_time: {exact_time*1e3:.1f} ms")

    return ValidationResult(
        polygon_family=polygon_family,
        seed=seed,
        n_cells=n_cells,
        n_strips=r,
        phi_deg=phi_star,
        heuristic_energy_J=heuristic_table_energy,
        optimal_energy_J=opt_energy,
        gap_pct=gap_pct,
        heuristic_time_s=heuristic_time,
        exact_time_s=exact_time,
        wind_speed_mps=wind_speed,
        wind_dir_deg=wind_dir_deg,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exact vs heuristic validation")
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--cell-size', type=float, default=30.0,
                        help='Cell size in metres (default 30)')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of random seeds per config (default 5)')
    args = parser.parse_args()

    pm = DEFAULT_POWER_MODEL
    cell_w = cell_h = args.cell_size
    CONFIGS = [
        # (family, wind_dir_deg, wind_speed)
        ("convex",       0,   3.0),
        ("convex",      90,   3.0),
        ("mild_concave", 0,   3.0),
        ("mild_concave", 45,  3.6),
        ("high_concave", 90,  2.4),
        ("corridor",     0,   3.0),
        ("corridor",    45,   3.6),
    ]

    results: List[ValidationResult] = []

    print(f"Exact (Held-Karp) vs heuristic validation")
    print(f"Cell size: {cell_w}×{cell_h} m")
    print(f"Power model: Zeng et al. (P0={pm.P0}, Pi={pm.Pi}, "
          f"Utip={pm.U_tip}, v0={pm.v0})")
    print()

    for family, wind_dir, wind_speed in CONFIGS:
        if args.verbose:
            print(f"=== {family}  wind={wind_speed:.1f} m/s @ {wind_dir}°  "
                  f"cell={cell_w}m ===")
        for seed in range(args.seeds):
            res = run_validation(
                family, seed, cell_w, cell_h,
                wind_dir, wind_speed, pm,
                verbose=args.verbose,
            )
            if res is not None:
                results.append(res)
                if not args.verbose:
                    gap_s = f"{res.gap_pct:+.2f}%"
                    print(f"  {family:15s}  seed={seed}  r={res.n_strips:2d}  "
                          f"cells={res.n_cells:3d}  φ*={res.phi_deg:5.0f}°  "
                          f"gap={gap_s:>8s}  "
                          f"exact_ms={res.exact_time_s*1000:6.1f}")

    if not results:
        print("No results generated.")
        return

    # Summary statistics
    gaps = [r.gap_pct for r in results]
    strip_counts = [r.n_strips for r in results]
    print()
    print("=" * 70)
    print(f"Results: {len(results)} instances")
    print(f"  strips: min={min(strip_counts)}  max={max(strip_counts)}  "
          f"mean={np.mean(strip_counts):.1f}")
    print(f"  gap(%): min={min(gaps):.3f}  max={max(gaps):.3f}  "
          f"mean={np.mean(gaps):.3f}  median={np.median(gaps):.3f}")
    n_opt = sum(1 for g in gaps if abs(g) < 0.01)
    print(f"  heuristic matches exact (gap < 0.01%): {n_opt}/{len(results)}")
    n_improves = sum(1 for g in gaps if g > 0.01)
    if n_improves > 0:
        imp_gaps = [g for g in gaps if g > 0.01]
        print(f"  instances where exact improves: {n_improves}  "
              f"(mean gap when improved: {np.mean(imp_gaps):.2f}%)")
    print()

    # Write CSV
    csv_path = DATA_DIR / "minlp_validation.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'polygon_family', 'seed', 'n_cells', 'n_strips', 'phi_deg',
            'wind_dir_deg', 'wind_speed_mps',
            'heuristic_energy_J', 'optimal_energy_J', 'gap_pct',
            'heuristic_time_s', 'exact_time_s',
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'polygon_family':   r.polygon_family,
                'seed':             r.seed,
                'n_cells':          r.n_cells,
                'n_strips':         r.n_strips,
                'phi_deg':          f"{r.phi_deg:.1f}",
                'wind_dir_deg':     f"{r.wind_dir_deg:.0f}",
                'wind_speed_mps':   f"{r.wind_speed_mps:.2f}",
                'heuristic_energy_J': f"{r.heuristic_energy_J:.2f}",
                'optimal_energy_J': f"{r.optimal_energy_J:.2f}",
                'gap_pct':          f"{r.gap_pct:.4f}",
                'heuristic_time_s': f"{r.heuristic_time_s:.4f}",
                'exact_time_s':     f"{r.exact_time_s:.6f}",
            })
    print(f"Results written to: {csv_path}")

    # Write summary text
    summary_path = DATA_DIR / "minlp_validation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("MINLP Validation: Exact (Held-Karp DP) vs Heuristic\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Power model (Zeng et al. 2019):\n")
        pm = DEFAULT_POWER_MODEL
        f.write(f"  P0 = {pm.P0} W,  Pi = {pm.Pi} W\n")
        f.write(f"  Utip = {pm.U_tip} m/s,  v0 = {pm.v0} m/s\n")
        f.write(f"  rho = {pm.rho} kg/m3,  C_D = {pm.C_D},  A = {pm.A} m2\n")
        f.write(f"\nInstances: {len(results)}\n")
        f.write(f"Strips per instance: min={min(strip_counts)} "
                f"max={max(strip_counts)} mean={np.mean(strip_counts):.1f}\n")
        f.write(f"\nEnergy gap (heuristic − exact) / exact × 100:\n")
        f.write(f"  min  = {min(gaps):.4f}%\n")
        f.write(f"  max  = {max(gaps):.4f}%\n")
        f.write(f"  mean = {np.mean(gaps):.4f}%\n")
        f.write(f"  std  = {np.std(gaps):.4f}%\n")
        f.write(f"\nInstances where heuristic matches optimal (gap < 0.01%): "
                f"{n_opt}/{len(results)}\n")
        if n_improves > 0:
            f.write(f"Instances where exact solver improves on heuristic: "
                    f"{n_improves}/{len(results)}\n")
            f.write(f"  Mean improvement when improved: "
                    f"{np.mean([g for g in gaps if g > 0.01]):.2f}%\n")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
