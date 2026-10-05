"""
farmland_kml.py
===============
Run the energy-optimal planner for all six Farmland field-test configurations
and export:
  - Google Earth KML  (paper/data/farmland_paths/<tag>.kml)
  - Mission Planner   (paper/data/farmland_paths/<tag>.waypoints)
  - JSON summary      (paper/data/farmland_paths/results.json)

Usage
-----
    python code/farmland_kml.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "gcs_plugin"))

from decomposition import decompose
from routing import build_route, optimal_grid_angle_energy, DEFAULT_POWER_MODEL
from gcs_plugin.coords import (
    parse_kml_polygon,
    lonlat_to_utm_polygon,
    from_utm,
)
from gcs_plugin.camera import camera_from_preset
from gcs_plugin.export import write_kml, write_waypoints
from routing import RoutingResult

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

KML_SCENARIO = Path(__file__).parent.parent / "code" / "scenarios" / "Farmland.kml"
OUT_DIR = Path(__file__).parent.parent / "paper" / "data" / "farmland_paths"

# ---------------------------------------------------------------------------
# Runs: (tag, description, alt_m, overlap_side, overlap_front,
#        wind_dir_from_deg, wind_speed_ms, nominal_speed_ms)
# ---------------------------------------------------------------------------

RUNS = [
    ("farmland_alt50_ov70",       "Low alt. (50 m)", 50,  0.70, 0.80, 45,  3.0, 8.0),
    ("farmland_alt80_ov70_baseline", "Baseline (80 m)", 80,  0.70, 0.80, 45,  3.0, 8.0),
    ("farmland_alt120_ov70",      "High alt. (120 m)", 120, 0.70, 0.80, 45,  3.0, 8.0),
    ("farmland_alt80_ov80_dense", "Dense (80 m)", 80,  0.80, 0.85, 45,  3.0, 8.0),
    ("farmland_alt80_ov60_rapid", "Rapid (80 m)", 80,  0.60, 0.70, 45,  3.0, 8.0),
    ("farmland_alt80_wind180_6ms","Wind 180 deg / 6 m/s", 80,  0.70, 0.80, 180, 6.0, 10.0),
]


def route_to_leg_speeds(
    route: RoutingResult,
    speed_precision: float = 0.5,
) -> tuple[list, float | None]:
    """Convert route.leg_ground_speeds_mps to the (start_idx, end_idx, speed_ms)
    tuples expected by write_waypoints, skipping zero-length duplicate waypoints
    that appear at each inter-strip transition.  Speeds are rounded to
    *speed_precision* m/s to reduce the number of DO_CHANGE_SPEED commands.

    Groundspeeds (not airspeeds) are used because export.py writes
    MAVLink DO_CHANGE_SPEED with speed_type=1 (groundspeed).

    Returns (leg_speeds, transition_speed_ms) or (None, None) if no speed data.
    """
    if not route.leg_ground_speeds_mps:
        return None, None

    ti_set = set(route.transition_indices)

    # Build seg_start → rounded_speed, skipping zero-length duplicate segments.
    seg_speed: dict[int, float] = {}
    speed_idx = 0
    for seg in range(len(route.waypoints) - 1):
        if seg in ti_set:
            continue
        raw = route.leg_ground_speeds_mps[speed_idx]
        seg_speed[seg] = round(raw / speed_precision) * speed_precision
        speed_idx += 1

    # Median speed of inter-strip transition segments (seg = ti - 1).
    trans_speeds = [
        seg_speed[ti - 1] for ti in route.transition_indices if (ti - 1) in seg_speed
    ]
    transition_speed_ms = (
        float(np.median(trans_speeds)) if trans_speeds else None
    )

    # Override ALL transition segments to exactly transition_speed_ms so that
    # write_waypoints can reliably detect them via its speed-equality check
    # (tolerance 0.01 m/s).  Using the raw median without this override caused
    # only a subset of transitions to be detected when airspeeds varied.
    if transition_speed_ms is not None:
        for ti in route.transition_indices:
            if (ti - 1) in seg_speed:
                seg_speed[ti - 1] = transition_speed_ms

    # Group runs of identical speed into (start_wp, end_wp, speed) tuples.
    leg_speeds: list[tuple[int, int, float]] = []
    prev_speed: float | None = None
    run_start: int | None = None
    for seg in sorted(seg_speed):
        spd = seg_speed[seg]
        if spd != prev_speed:
            if run_start is not None:
                leg_speeds.append((run_start, seg, prev_speed))  # type: ignore[arg-type]
            run_start = seg
            prev_speed = spd
    if run_start is not None:
        leg_speeds.append((run_start, len(route.waypoints) - 1, prev_speed))  # type: ignore[arg-type]

    return leg_speeds, transition_speed_ms


def met_to_wind_vec(wind_dir_from_deg: float, wind_speed_ms: float) -> np.ndarray:
    toward_deg = (wind_dir_from_deg + 180.0) % 360.0
    rad = math.radians(90.0 - toward_deg)
    return np.array([math.cos(rad), math.sin(rad)]) * wind_speed_ms


def run_and_export(
    tag: str,
    description: str,
    alt: float,
    overlap_side: float,
    overlap_front: float,
    wind_dir_deg: float,
    wind_speed_ms: float,
    nominal_speed_ms: float,
    lonlat_polygon: list,
    polygon_utm,
    crs,
    out_dir: Path,
) -> dict:
    cp = camera_from_preset("sony_a6000_16mm", alt, overlap_side, overlap_front)
    wind_vec = met_to_wind_vec(wind_dir_deg, wind_speed_ms)

    phi_deg = optimal_grid_angle_energy(
        polygon_utm,
        cp.cell_w,
        cp.cell_h,
        wind_vec,
        nominal_speed_ms,
        power_model=DEFAULT_POWER_MODEL,
    )
    decomp = decompose(polygon_utm, cp.cell_w, cp.cell_h, angle_deg=phi_deg)
    route = build_route(
        decomp,
        wind_vec,
        airspeed=nominal_speed_ms,
        power_model=DEFAULT_POWER_MODEL,
        objective="energy",
    )

    # Convert UTM waypoints → (lon, lat)
    lonlat_wps = from_utm(route.waypoints, crs)

    # Write KML
    kml_path = out_dir / f"{tag}.kml"
    write_kml(
        kml_path,
        lonlat_wps,
        altitude_m=alt,
        survey_polygon=lonlat_polygon,
        name=description,
        description=(
            f"Alt {alt} m | Overlap {int(overlap_side*100)}/{int(overlap_front*100)}% "
            f"| Wind {wind_dir_deg:.0f} deg at {wind_speed_ms:.1f} m/s "
            f"| Speed {nominal_speed_ms:.1f} m/s | phi*={phi_deg:.1f} deg"
        ),
    )

    # Write Mission Planner waypoints (with per-leg DO_CHANGE_SPEED commands)
    wp_path = out_dir / f"{tag}.waypoints"
    leg_speeds, transition_speed_ms = route_to_leg_speeds(route)
    write_waypoints(
        wp_path,
        lonlat_wps,
        altitude_m=alt,
        trigger_distance_m=cp.trigger_distance_m,
        leg_speeds=leg_speeds,
        transition_speed_ms=transition_speed_ms,
        trigger_waypoints=frozenset(route.transition_indices),
    )

    metrics = {
        "tag": tag,
        "description": description,
        "altitude_m": alt,
        "overlap_side_pct": int(round(overlap_side * 100)),
        "overlap_front_pct": int(round(overlap_front * 100)),
        "wind_dir_deg": wind_dir_deg,
        "wind_speed_ms": wind_speed_ms,
        "nominal_speed_ms": nominal_speed_ms,
        "phi_opt_deg": round(float(phi_deg), 1),
        "n_cells": int(decomp.n_cells),
        "n_strips": int(route.metrics["n_strips"]),
        "n_waypoints": len(lonlat_wps),
        "path_km": round(float(route.total_length_m / 1000.0), 3),
        "est_time_min": round(float(route.total_time_s / 60.0), 2),
        "energy_kJ": round(float(route.total_energy_J / 1000.0), 2),
        "cell_w_m": round(float(cp.cell_w), 2),
        "cell_h_m": round(float(cp.cell_h), 2),
        "trigger_distance_m": round(float(cp.trigger_distance_m), 2),
    }

    print(
        f"  {tag}: phi*={phi_deg:.1f} deg | {metrics['n_strips']} strips | "
        f"{metrics['path_km']:.2f} km | {metrics['energy_kJ']:.1f} kJ"
    )
    return metrics


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse scenario polygon once
    lonlat_polygon = parse_kml_polygon(str(KML_SCENARIO))
    polygon_utm, crs = lonlat_to_utm_polygon(lonlat_polygon)

    results = []
    for run_spec in RUNS:
        tag, description, alt, os, of, wd, ws, gs = run_spec
        print(f"Running {tag} ...")
        metrics = run_and_export(
            tag, description, alt, os, of, wd, ws, gs,
            lonlat_polygon, polygon_utm, crs, OUT_DIR,
        )
        results.append(metrics)

    # Save JSON summary
    json_path = OUT_DIR / "results.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} runs to {OUT_DIR}")
    print(f"  KML + waypoints: {OUT_DIR}/*.kml, *.waypoints")
    print(f"  Summary:         {json_path}")


if __name__ == "__main__":
    main()
