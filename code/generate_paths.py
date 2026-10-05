"""
generate_paths.py
=================
Generate and compare coverage paths for a given scenario:
  - Proposed method: wind-aligned selective grid (any-intersection rule)
  - Baseline 1:      fixed-orientation (0°) selective grid
  - Baseline 2:      wind-aligned non-pruned grid
                     (same cell size, but retain ALL cells inside bounding box)

Records per-run metrics to paper/data/ and calls visualize.py to produce
paper/figures/fig_coverage_paths.pdf and paper/figures/fig_wind_sweep.pdf.

Usage
-----
    python generate_paths.py                     # uses scenario_01.json
    python generate_paths.py --scenario 01
    python generate_paths.py --wind-sweep        # sweep all 8 wind directions
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from instance_gen import load_scenario_polygon
from decomposition import decompose, DecompositionResult
from routing import build_route, RoutingResult, optimal_grid_angle, optimal_grid_angle_energy, DEFAULT_POWER_MODEL
from visualize import (
    figure_coverage_paths,
    figure_wind_sweep,
    quick_plot,
    FIGURES_DIR,
)

DATA_DIR = Path(__file__).parent.parent / "paper" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Baseline 2 helper: non-pruned grid (all cells in bounding box)
# ---------------------------------------------------------------------------

def decompose_nonpruned(polygon, cell_w, cell_h, angle_deg):
    """
    Non-pruned grid baseline: retain every cell in the bounding box,
    regardless of whether it intersects P.  Used purely as a comparison baseline.
    Note: coverage_ratio may exceed 1 (overshoot on all sides).
    """
    import math
    from shapely.affinity import rotate
    from shapely.geometry import box
    from decomposition import Cell, DecompositionResult
    import networkx as nx

    if not polygon.is_valid:
        from shapely.validation import make_valid
        polygon = make_valid(polygon)

    result = DecompositionResult(
        polygon=polygon, angle_deg=angle_deg,
        cell_w=cell_w, cell_h=cell_h,
    )

    cx, cy = polygon.centroid.x, polygon.centroid.y
    poly_rot = rotate(polygon, -angle_deg, origin=(cx, cy), use_radians=False)
    minx, miny, maxx, maxy = poly_rot.bounds

    origin_x = math.floor(minx / cell_w) * cell_w
    origin_y = math.floor(miny / cell_h) * cell_h
    col_min = int(math.floor((minx - origin_x) / cell_w))
    col_max = int(math.ceil((maxx - origin_x) / cell_w))
    row_min = int(math.floor((miny - origin_y) / cell_h))
    row_max = int(math.ceil((maxy - origin_y) / cell_h))

    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            x0 = origin_x + col * cell_w
            y0 = origin_y + row * cell_h
            cell_rot = box(x0, y0, x0 + cell_w, y0 + cell_h)
            cell_orig = rotate(cell_rot, angle_deg, origin=(cx, cy), use_radians=False)
            inter = cell_rot.intersection(poly_rot)
            inter_area = inter.area if not inter.is_empty else 0.0
            is_boundary = (inter_area < cell_rot.area - 1e-9) and (inter_area > 0)
            centroid = np.array([cell_orig.centroid.x, cell_orig.centroid.y])
            cell_obj = Cell(
                row=row, col=col, footprint=cell_orig, is_boundary=is_boundary,
                center=centroid, intersection_area=inter_area,
            )
            idx = len(result.cells)
            result.cells.append(cell_obj)
            result.index_map[(row, col)] = idx

    G = nx.Graph()
    G.add_nodes_from(range(result.n_cells))
    for (row, col), idx in result.index_map.items():
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nb = (row + dr, col + dc)
            if nb in result.index_map:
                nb_idx = result.index_map[nb]
                if not G.has_edge(idx, nb_idx):
                    G.add_edge(idx, nb_idx)
    result.adjacency = G
    return result


# ---------------------------------------------------------------------------
# Run one configuration and time it
# ---------------------------------------------------------------------------

def run_config(polygon, cell_w, cell_h, angle_deg, wind_vec, airspeed,
               nonpruned: bool = False,
               objective: str = "energy") -> tuple:
    t0 = time.perf_counter()
    if nonpruned:
        decomp = decompose_nonpruned(polygon, cell_w, cell_h, angle_deg)
    else:
        decomp = decompose(polygon, cell_w, cell_h, angle_deg)
    route = build_route(decomp, wind_vec=wind_vec, airspeed=airspeed,
                        power_model=DEFAULT_POWER_MODEL, objective=objective)
    runtime = time.perf_counter() - t0
    return decomp, route, runtime


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate and compare coverage paths")
    parser.add_argument("--scenario", default="01",
                         help="Scenario ID (default: 01)")
    parser.add_argument("--wind-sweep", action="store_true",
                         help="Sweep all 8 wind directions and produce wind-sweep figure")
    args = parser.parse_args()

    scenario_path = Path(__file__).parent / "scenarios" / f"scenario_{args.scenario}.json"
    if not scenario_path.exists():
        print(f"Error: scenario file not found: {scenario_path}")
        sys.exit(1)

    with open(scenario_path) as f:
        cfg = json.load(f)

    print(f"Loading scenario: {scenario_path}")
    poly = load_scenario_polygon(str(scenario_path))
    wind_deg = cfg["wind"]["direction_deg"]
    wind_speed = cfg["wind"]["speed_mps"]
    airspeed = cfg["uav"]["airspeed_mps"]
    cell_w = cfg["cell"]["width_m"]
    cell_h = cfg["cell"]["height_m"]

    def make_wind(deg, spd):
        rad = math.radians(deg)
        return np.array([spd * math.cos(rad), spd * math.sin(rad)])

    wind_vec = make_wind(wind_deg, wind_speed)
    print(f"Wind: {wind_speed} m/s at {wind_deg}°, airspeed={airspeed} m/s, "
          f"cell={cell_w}×{cell_h} m")

    # ------------------------------------------------------------------
    # Generate three configurations
    # ------------------------------------------------------------------
    print("\n--- Proposed: energy-optimal selective grid ---")
    phi_prop = optimal_grid_angle_energy(poly, cell_w, cell_h, wind_vec, airspeed,
                                          DEFAULT_POWER_MODEL)
    print(f"  energy-optimal phi = {phi_prop:.1f}°")
    d_prop, r_prop, rt_prop = run_config(poly, cell_w, cell_h, phi_prop, wind_vec, airspeed,
                                          objective="energy")
    print(f"  cells={d_prop.n_cells}, energy={r_prop.total_energy_J/1000.0:.1f}kJ, time={r_prop.total_time_s:.0f}s, "
          f"length={r_prop.total_length_m:.0f}m, turns={r_prop.turn_count}, "
          f"runtime={rt_prop*1000:.1f}ms, coverage={d_prop.coverage_ratio:.6f}")

    print("--- Baseline 1: fixed 0° selective grid ---")
    d_b1, r_b1, rt_b1 = run_config(poly, cell_w, cell_h, 0.0, wind_vec, airspeed,
                                     objective="time")
    print(f"  cells={d_b1.n_cells}, energy={r_b1.total_energy_J/1000.0:.1f}kJ, time={r_b1.total_time_s:.0f}s, "
          f"length={r_b1.total_length_m:.0f}m, turns={r_b1.turn_count}, "
          f"runtime={rt_b1*1000:.1f}ms, coverage={d_b1.coverage_ratio:.6f}")

    print("--- Baseline 2: time-optimal orientation, selective grid ---")
    phi_b2 = optimal_grid_angle(poly, cell_w, cell_h, wind_vec, airspeed)
    print(f"  time-optimal phi = {phi_b2:.1f}°")
    d_b2, r_b2, rt_b2 = run_config(poly, cell_w, cell_h, phi_b2, wind_vec, airspeed,
                                     objective="time")
    print(f"  cells={d_b2.n_cells}, energy={r_b2.total_energy_J/1000.0:.1f}kJ, time={r_b2.total_time_s:.0f}s, "
          f"length={r_b2.total_length_m:.0f}m, turns={r_b2.turn_count}, "
          f"runtime={rt_b2*1000:.1f}ms, coverage={d_b2.coverage_ratio:.6f}")

    print("--- Baseline 3: fixed 0° orientation, energy-optimal speed ---")
    d_b3, r_b3, rt_b3 = run_config(poly, cell_w, cell_h, 0.0, wind_vec, airspeed,
                                     objective="energy")
    print(f"  cells={d_b3.n_cells}, energy={r_b3.total_energy_J/1000.0:.1f}kJ, time={r_b3.total_time_s:.0f}s, "
          f"length={r_b3.total_length_m:.0f}m, turns={r_b3.turn_count}, "
          f"runtime={rt_b3*1000:.1f}ms, coverage={d_b3.coverage_ratio:.6f}")

    # ------------------------------------------------------------------
    # Save metrics CSV
    # ------------------------------------------------------------------
    csv_path = DATA_DIR / f"scenario_{args.scenario}_comparison.csv"
    rows = [
        dict(method="proposed", angle_deg=phi_prop, n_cells=d_prop.n_cells,
             n_boundary=d_prop.n_boundary_cells, coverage_ratio=d_prop.coverage_ratio,
               energy_kJ=r_prop.total_energy_J / 1000.0,
             flight_time_s=r_prop.total_time_s, path_length_m=r_prop.total_length_m,
             turns=r_prop.turn_count, runtime_s=rt_prop),
        dict(method="baseline1_fixed0", angle_deg=0.0, n_cells=d_b1.n_cells,
             n_boundary=d_b1.n_boundary_cells, coverage_ratio=d_b1.coverage_ratio,
               energy_kJ=r_b1.total_energy_J / 1000.0,
             flight_time_s=r_b1.total_time_s, path_length_m=r_b1.total_length_m,
             turns=r_b1.turn_count, runtime_s=rt_b1),
        dict(method="baseline2_time_phi", angle_deg=phi_b2, n_cells=d_b2.n_cells,
             n_boundary=d_b2.n_boundary_cells, coverage_ratio=d_b2.coverage_ratio,
               energy_kJ=r_b2.total_energy_J / 1000.0,
             flight_time_s=r_b2.total_time_s, path_length_m=r_b2.total_length_m,
             turns=r_b2.turn_count, runtime_s=rt_b2),
        dict(method="baseline3_fixed0_energy", angle_deg=0.0, n_cells=d_b3.n_cells,
             n_boundary=d_b3.n_boundary_cells, coverage_ratio=d_b3.coverage_ratio,
               energy_kJ=r_b3.total_energy_J / 1000.0,
             flight_time_s=r_b3.total_time_s, path_length_m=r_b3.total_length_m,
             turns=r_b3.turn_count, runtime_s=rt_b3),
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved comparison CSV: {csv_path}")

    # ------------------------------------------------------------------
    # Figure 1: three-panel coverage path  (named per scenario)
    # ------------------------------------------------------------------
    scene_tag = cfg.get("display_name", f"scenario_{args.scenario}").lower().replace(" ", "_")
    fig_path = FIGURES_DIR / f"fig_coverage_paths_{scene_tag}.pdf"
    figure_coverage_paths(
        poly, d_prop, r_prop, d_b1, r_b1, d_b2, r_b2,
        wind_vec=wind_vec,
        out_path=fig_path,
        baseline3_decomp=d_b3, baseline3_route=r_b3,
    )
    # Also keep the default name for the first / canonical scenario so existing
    # paper references (fig_coverage_paths.pdf) continue to work.
    if args.scenario == "01":
        figure_coverage_paths(
            poly, d_prop, r_prop, d_b1, r_b1, d_b2, r_b2,
            wind_vec=wind_vec,
            out_path=FIGURES_DIR / "fig_coverage_paths.pdf",
            baseline3_decomp=d_b3, baseline3_route=r_b3,
        )

    # ------------------------------------------------------------------
    # Optional: wind direction sweep
    # ------------------------------------------------------------------
    if args.wind_sweep:
        print("\n--- Wind direction sweep ---")
        sweep_dirs = [0, 45, 90, 135, 180, 225, 270, 315]
        proposed_energy, baseline_energy = [], []
        sweep_rows = []

        for wd in sweep_dirs:
            wv = make_wind(wd, wind_speed)
            phi_wd = optimal_grid_angle_energy(poly, cell_w, cell_h, wv, airspeed,
                                               DEFAULT_POWER_MODEL)
            _, rp, _ = run_config(poly, cell_w, cell_h, phi_wd, wv, airspeed,
                                  objective="energy")
            _, rb, _ = run_config(poly, cell_w, cell_h, 0.0, wv, airspeed,
                                  objective="time")
            proposed_energy.append(rp.total_energy_J / 1000.0)
            baseline_energy.append(rb.total_energy_J / 1000.0)
            sweep_rows.append(dict(
                wind_dir_deg=wd, proposed_time_s=rp.total_time_s,
                baseline_time_s=rb.total_time_s,
                proposed_energy_kJ=rp.total_energy_J / 1000.0,
                baseline_energy_kJ=rb.total_energy_J / 1000.0,
                delta_s=rb.total_time_s - rp.total_time_s,
            ))
            print(f"  dir={wd:3d}°  proposed={rp.total_time_s:.0f}s  "
                  f"baseline={rb.total_time_s:.0f}s  "
                  f"delta={rb.total_time_s-rp.total_time_s:+.0f}s")

        # No-wind reference
        wv0 = make_wind(0, 0.0)
        _, r0, _ = run_config(poly, cell_w, cell_h, 0.0, wv0, airspeed)
        nowind_energy = r0.total_energy_J / 1000.0

        sweep_csv = DATA_DIR / "wind_sweep_scenario01.csv"
        with open(sweep_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sweep_rows[0].keys())
            writer.writeheader()
            writer.writerows(sweep_rows)
        print(f"Saved wind sweep CSV: {sweep_csv}")

        # Convert code math (toward) angles to aviation FROM bearings and sort
        from_bearings = [(270 - d) % 360 for d in sweep_dirs]
        combined = sorted(zip(from_bearings, proposed_energy, baseline_energy))
        sorted_from, sorted_proposed, sorted_baseline = zip(*combined)
        figure_wind_sweep(list(sorted_from), list(sorted_proposed), list(sorted_baseline),
                          nowind_energy,
                          out_path=FIGURES_DIR / "fig_wind_sweep.pdf")

    print("\nDone.")


if __name__ == "__main__":
    main()
