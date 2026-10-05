"""
planner.py
==========
Wind-aware coverage path planner for UAV survey missions.

Usage
-----
    python3 code/gcs_plugin/planner.py <polygon.kml> [options]

The polygon should be drawn in Google Earth and saved as a KML file.
The script projects the polygon to a local UTM frame, runs the selective
grid decomposition and strip routing, and exports:
  (1) A KML file loadable in Google Earth showing the flight path and waypoints.
  (2) A .waypoints file loadable in Mission Planner.

Examples
--------
    # Sony A6000 at 80 m, 70% side / 80% front overlap, NE wind 5 m/s
    python3 code/gcs_plugin/planner.py field.kml \\
        --wind-dir 45 --wind-speed 5 --altitude 80 \\
        --camera sony_a6000_16mm --overlap-side 0.70 --overlap-front 0.80 \\
        --ground-speed 8 \\
        --output-kml flight_path.kml --output-wp flight_path.waypoints

    # Custom FOV
    python3 code/gcs_plugin/planner.py field.kml \\
        --wind-dir 180 --wind-speed 3 --altitude 60 \\
        --fov-h 84 --fov-v 70 --overlap-side 0.65 --overlap-front 0.75 \\
        --ground-speed 10 --output-kml out.kml --output-wp out.waypoints

Wind direction convention
-------------------------
Meteorological: direction the wind comes *from*, measured clockwise from North.
e.g. --wind-dir 45 means NE wind (wind blows toward SW).
Internally the planner uses the direction the wind blows *toward*
(= wind_dir + 180° mod 360°) to compute speed benefit.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

# Make sure the parent code/ directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decomposition import decompose
from routing import build_route, optimal_grid_angle, optimal_grid_angle_energy, DEFAULT_POWER_MODEL
from gcs_plugin.camera import camera_from_preset, camera_from_fov, SENSOR_PRESETS
from gcs_plugin.coords import parse_kml_polygon, lonlat_to_utm_polygon, from_utm
from gcs_plugin.export import write_waypoints, write_kml


# ---------------------------------------------------------------------------
# Wind vector helpers
# ---------------------------------------------------------------------------

def met_to_wind_vec(wind_dir_from_deg: float, wind_speed_ms: float) -> np.ndarray:
    """
    Convert meteorological wind (direction wind comes FROM, CW from North)
    to a Cartesian wind vector [vwx, vwy] pointing in the direction the air moves.

    Convention used in decomposition/routing:
        x → East,  y → North
    """
    # Direction wind blows TOWARD (meteorological "from" + 180°)
    toward_deg = (wind_dir_from_deg + 180.0) % 360.0
    # Convert to math angle (CCW from East)
    math_deg = 90.0 - toward_deg
    rad = math.radians(math_deg)
    return np.array([wind_speed_ms * math.cos(rad), wind_speed_ms * math.sin(rad)])


def wind_grid_angle(wind_dir_from_deg: float) -> float:
    """
    Grid orientation angle phi for wind-aligned decomposition.

    The strip axis is parallel to the wind direction.  phi is measured
    CCW from the East axis (= the x-axis of the UTM/rotated frame).
    """
    # Wind blows toward direction:
    toward_deg = (wind_dir_from_deg + 180.0) % 360.0
    # Convert North-CW to East-CCW
    phi = (90.0 - toward_deg) % 180.0
    return phi


# ---------------------------------------------------------------------------
# Main planner function (callable from scripts or other modules)
# ---------------------------------------------------------------------------

def plan(
    kml_path: str | Path,
    wind_dir_from_deg: float,
    wind_speed_ms: float,
    altitude_m: float,
    cell_w: float,
    cell_h: float,
    ground_speed_ms: float = 8.0,
    trigger_distance_m: float = 0.0,
    overlap_side: float = 0.0,
    overlap_front: float = 0.0,
    optimize_speed: bool = True,
    min_groundspeed_ms: float = 8.0,
    max_groundspeed_ms: float = 25.0,
    transition_speed_ms: float | None = None,
    camera_fps: float | None = None,
    output_kml: str | Path | None = None,
    output_wp: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run the full planning pipeline for a KML-defined polygon.

    Parameters
    ----------
    kml_path             : path to Google Earth KML file containing the survey polygon
    wind_dir_from_deg    : meteorological wind direction (degrees, from North CW)
    wind_speed_ms        : wind speed in m/s
    altitude_m           : flight altitude AGL in metres
    cell_w               : cell width in metres (cross-track, from camera footprint)
    cell_h               : cell height in metres (along-track, from camera footprint)
    ground_speed_ms      : nominal cruise ground speed in m/s; used as fallback when
                           optimize_speed=False.  Must exceed wind_speed_ms.
    trigger_distance_m   : camera trigger distance in metres (0 = no trigger command)
    overlap_side         : lateral overlap fraction (written to KML metadata)
    overlap_front        : along-track overlap fraction (written to KML metadata)
    optimize_speed       : if True, optimize per-leg airspeed using the power model
    min_groundspeed_ms   : lower speed bound used for optimisation/export (m/s)
    max_groundspeed_ms   : upper speed bound used for optimisation/export (m/s).
                           Automatically clamped to camera_fps*cell_h if camera_fps given.
    transition_speed_ms  : optional fixed safety groundspeed for inter-strip transitions (m/s);
                           None leaves transition speeds energy-optimized.
    camera_fps           : camera burst frame rate (fps); used to clamp max_groundspeed_ms
                           so the trigger interval stays within camera capability.
                           Pass None (default) to leave max_groundspeed_ms unconstrained.
    output_kml           : path to write Google Earth KML output (None = skip)
    output_wp            : path to write Mission Planner .waypoints (None = skip)
    verbose              : print progress to stdout

    Returns
    -------
    dict with keys: n_cells, n_strips, total_time_s, total_length_m,
                    turn_count, n_waypoints, crs_epsg, mean_strip_speed_ms
    """
    # ---- 1. Parse polygon from KML ----
    if verbose:
        print(f"Reading polygon from: {kml_path}")
    lonlat_verts = parse_kml_polygon(kml_path)
    polygon_utm, crs = lonlat_to_utm_polygon(lonlat_verts)

    if verbose:
        area_ha = polygon_utm.area / 10_000
        print(f"  Polygon: {len(lonlat_verts)} vertices, area = {area_ha:.2f} ha")
        print(f"  UTM CRS: EPSG:{crs.to_epsg()}")

    # ---- 1b. Clamp max_groundspeed_ms to camera capability ----
    if camera_fps is not None and trigger_distance_m > 0:
        cam_max_vg = trigger_distance_m * camera_fps
        if cam_max_vg < max_groundspeed_ms:
            if verbose:
                print(
                    f"  Camera fps={camera_fps:.0f} constrains max groundspeed to "
                    f"{cam_max_vg:.1f} m/s (trigger interval = "
                    f"{trigger_distance_m/cam_max_vg:.2f} s)."
                )
            max_groundspeed_ms = cam_max_vg

    if max_groundspeed_ms < min_groundspeed_ms:
        raise ValueError(
            f"Maximum speed bound ({max_groundspeed_ms:.1f} m/s) is below "
            f"minimum speed bound ({min_groundspeed_ms:.1f} m/s)."
        )

    bounded_power_model = replace(
        DEFAULT_POWER_MODEL,
        v_a_min=min_groundspeed_ms,
        v_a_max=max_groundspeed_ms,
    )

    # ---- 2. Compute wind vector and find optimal grid orientation ----
    wind_vec = met_to_wind_vec(wind_dir_from_deg, wind_speed_ms)
    if optimize_speed:
        phi_deg = optimal_grid_angle_energy(
            polygon_utm, cell_w, cell_h, wind_vec,
            ground_speed_ms, bounded_power_model,
        )
    else:
        phi_deg = optimal_grid_angle(polygon_utm, cell_w, cell_h, wind_vec,
                                      ground_speed_ms)

    if verbose:
        print(f"  Wind: {wind_dir_from_deg:.0f}° from, {wind_speed_ms:.1f} m/s  "
              f"→  optimal grid phi = {phi_deg:.1f}°")
        print(f"  Cell size: {cell_w:.1f} m × {cell_h:.1f} m")

    # ---- 3. Decompose ----
    decomp = decompose(polygon_utm, cell_w, cell_h, angle_deg=phi_deg)
    if verbose:
        print(f"  Retained cells: {len(decomp.cells)}")

    if len(decomp.cells) == 0:
        raise RuntimeError(
            "No cells retained — check that cell size is smaller than the polygon."
        )

    # ---- 4. Route (with optional per-leg speed optimisation) ----
    if optimize_speed:
        route = build_route(
            decomp, wind_vec,
            airspeed=ground_speed_ms,
            power_model=bounded_power_model,
            objective="energy",
        )
        # Build per-leg speed list from routing result.
        # transition_indices stores the DESTINATION waypoint index (ti) of
        # each inter-strip transition.  The actual transition segment ends at
        # waypoints[ti], i.e. it is segment (ti-1).  A zero-length duplicate
        # waypoint also exists at each ti (waypoints[ti] repeated), so we must
        # skip those duplicates when consuming entries from leg_ground_speeds_mps.
        transition_set = set(route.transition_indices)
        n_legs = len(route.waypoints) - 1
        speed_idx = 0
        speed_precision_ms = 0.5
        segment_speeds: dict[int, float] = {}
        for leg_i in range(n_legs):
            if leg_i in transition_set:
                # Zero-length duplicate waypoint: no flown segment, no speed command.
                continue
            raw_vg = (route.leg_ground_speeds_mps[speed_idx]
                      if speed_idx < len(route.leg_ground_speeds_mps)
                      else ground_speed_ms)
            speed_idx += 1
            if (leg_i + 1) in transition_set and transition_speed_ms is not None:
                # Optional transition override for operational safety
                spd = transition_speed_ms
            else:
                spd = max(min_groundspeed_ms, min(max_groundspeed_ms, raw_vg))
            segment_speeds[leg_i] = round(spd / speed_precision_ms) * speed_precision_ms

        # Collapse into contiguous runs of equal speed.  Break runs across
        # zero-length duplicate segments so commands are attached only to flown legs.
        leg_speeds_list: list[tuple[int, int, float]] = []
        run_start: int | None = None
        prev_seg: int | None = None
        prev_speed: float | None = None
        for seg in sorted(segment_speeds):
            spd = segment_speeds[seg]
            starts_new = (
                run_start is None
                or prev_seg is None
                or seg != prev_seg + 1
                or prev_speed is None
                or abs(spd - prev_speed) >= 0.01
            )
            if starts_new:
                if run_start is not None and prev_seg is not None and prev_speed is not None:
                    leg_speeds_list.append((run_start, prev_seg + 1, prev_speed))
                run_start = seg
                prev_speed = spd
            prev_seg = seg
        if run_start is not None and prev_seg is not None and prev_speed is not None:
            leg_speeds_list.append((run_start, prev_seg + 1, prev_speed))

        mean_strip_spd = float(np.mean([
            spd for seg, spd in segment_speeds.items()
            if (seg + 1) not in transition_set
        ])) if route.leg_ground_speeds_mps else ground_speed_ms
    else:
        route = build_route(decomp, wind_vec, airspeed=ground_speed_ms)
        leg_speeds_list = None
        mean_strip_spd = ground_speed_ms

    est_time_s = route.total_time_s

    if verbose:
        print(f"  Strips: {len(route.strips)}, "
              f"waypoints: {len(route.waypoints)}, "
              f"path length: {route.total_length_m/1000:.2f} km, "
              f"est. time: {est_time_s:.0f} s ({est_time_s/60:.1f} min)")
        if optimize_speed and leg_speeds_list:
            if transition_speed_ms is None:
                print(f"  Mean strip-leg groundspeed: {mean_strip_spd:.1f} m/s")
            else:
                print(f"  Mean strip-leg groundspeed: {mean_strip_spd:.1f} m/s "
                      f"(transition override: {transition_speed_ms:.1f} m/s)")

    # ---- 5. Convert waypoints to lon/lat ----
    wp_lonlat = from_utm(route.waypoints, crs)

    # ---- 6. Export ----
    if output_kml:
        write_kml(
            output_kml,
            wp_lonlat,
            altitude_m,
            survey_polygon=lonlat_verts,
            name=Path(kml_path).stem + " flight path",
            description=(
                f"Wind: {wind_dir_from_deg:.0f} deg from, {wind_speed_ms:.1f} m/s | "
                f"Alt: {altitude_m:.0f} m AGL | "
                f"Cell: {cell_w:.1f}x{cell_h:.1f} m | "
                f"Overlap: side {overlap_side*100:.0f}% / front {overlap_front*100:.0f}% | "
                f"Cells: {len(decomp.cells)} | "
                f"Path: {route.total_length_m/1000:.2f} km | "
                f"Est. time: {est_time_s:.0f} s ({est_time_s/60:.1f} min) | "
                f"Speed opt: {'yes' if optimize_speed else 'no'}"
            ),
        )
        if verbose:
            print(f"  KML output:        {output_kml}")

    if output_wp:
        write_waypoints(
            output_wp,
            wp_lonlat,
            altitude_m,
            trigger_distance_m=trigger_distance_m,
            leg_speeds=leg_speeds_list,
            transition_speed_ms=transition_speed_ms,
            trigger_waypoints=(
                frozenset(route.transition_indices) if leg_speeds_list is not None else None
            ),
        )
        if verbose:
            print(f"  Waypoints output:  {output_wp}")

    return {
        "n_cells":            len(decomp.cells),
        "n_strips":           len(route.strips),
        "total_time_s":       est_time_s,
        "total_length_m":     route.total_length_m,
        "turn_count":         route.turn_count,
        "n_waypoints":        len(route.waypoints),
        "crs_epsg":           crs.to_epsg(),
        "mean_strip_speed_ms": mean_strip_spd,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Wind-aware UAV coverage path planner.\n"
            "Reads a polygon from a Google Earth KML file and writes a Mission "
            "Planner .waypoints file and/or a Google Earth flight-path KML."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("kml", metavar="POLYGON.KML",
                   help="KML file containing the survey polygon (drawn in Google Earth).")

    # Wind
    p.add_argument("--wind-dir", type=float, required=True,
                   metavar="DEG",
                   help="Wind direction (meteorological: degrees FROM, CW from North).")
    p.add_argument("--wind-speed", type=float, required=True,
                   metavar="M_S",
                   help="Wind speed in m/s.")

    # Flight parameters
    p.add_argument("--altitude", type=float, required=True,
                   metavar="M",
                   help="Flight altitude AGL in metres.")
    p.add_argument("--ground-speed", type=float, default=8.0,
                   metavar="M_S",
                   help=("Nominal cruise ground speed in m/s (used when --no-optimize-speed "
                         "is set, or as a fallback). Must exceed wind speed (default: 8.0)."))

    # Speed optimisation
    p.add_argument("--no-optimize-speed", action="store_true",
                   help="Disable per-leg energy-optimal speed optimisation; use "
                        "--ground-speed for all route legs.")
    p.add_argument("--min-groundspeed", type=float, default=8.0,
                   metavar="M_S",
                   help="Lower speed bound used for optimisation and export (default: 8.0).")
    p.add_argument("--max-groundspeed", type=float, default=25.0,
                   metavar="M_S",
                   help=("Upper speed bound used for optimisation and export (default: 25.0). "
                         "Automatically clamped to camera fps × trigger_distance if "
                         "--camera-fps is given."))
    p.add_argument("--transition-speed", type=float, default=None,
                   metavar="M_S",
                   help=("Optional fixed safety groundspeed for inter-strip transitions. "
                         "Omit to export the energy-optimized transition speeds."))
    p.add_argument("--camera-fps", type=float, default=None,
                   metavar="FPS",
                   help="Camera burst frame rate (fps). If given, clamps --max-groundspeed "
                        "so the trigger interval stays within camera capability, and warns "
                        "if the energy-optimal speed would be too fast for the camera.")

    # Camera — either preset or manual FOV
    cam_group = p.add_mutually_exclusive_group(required=True)
    cam_group.add_argument("--camera",
                           choices=list(SENSOR_PRESETS),
                           metavar="PRESET",
                           help=(
                               f"Named camera preset. "
                               f"Available: {', '.join(SENSOR_PRESETS)}."
                           ))
    cam_group.add_argument("--fov-h", type=float, metavar="DEG",
                           help="Horizontal field-of-view in degrees (use with --fov-v).")
    p.add_argument("--fov-v", type=float, metavar="DEG",
                   help="Vertical field-of-view in degrees (required with --fov-h).")

    # Overlap
    p.add_argument("--overlap-side", type=float, default=0.70,
                   metavar="FRAC",
                   help="Lateral (cross-track) overlap fraction 0–1 (default: 0.70).")
    p.add_argument("--overlap-front", type=float, default=0.80,
                   metavar="FRAC",
                   help="Along-track overlap fraction 0–1 (default: 0.80).")

    # Output
    p.add_argument("--output-kml", metavar="FILE",
                   help="Output KML file path for Google Earth (default: <kml_stem>_path.kml).")
    p.add_argument("--output-wp", metavar="FILE",
                   help="Output .waypoints file path for Mission Planner "
                        "(default: <kml_stem>.waypoints).")
    p.add_argument("--no-trigger", action="store_true",
                   help="Omit camera trigger command from waypoint file.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output.")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate FOV arguments
    if args.fov_h is not None and args.fov_v is None:
        parser.error("--fov-v is required when --fov-h is used.")
    if args.fov_v is not None and args.fov_h is None:
        parser.error("--fov-h is required when --fov-v is used.")

    # Build camera footprint
    if args.camera:
        cam = camera_from_preset(
            args.camera, args.altitude,
            overlap_side=args.overlap_side,
            overlap_front=args.overlap_front,
        )
    else:
        cam = camera_from_fov(
            args.fov_h, args.fov_v, args.altitude,
            overlap_side=args.overlap_side,
            overlap_front=args.overlap_front,
        )

    if not args.quiet:
        print(f"Camera: {cam.summary()}")

    # Default output paths
    stem = Path(args.kml).stem
    out_kml = args.output_kml or f"{stem}_path.kml"
    out_wp  = args.output_wp  or f"{stem}.waypoints"

    trigger = 0.0 if args.no_trigger else cam.trigger_distance_m

    if args.ground_speed <= args.wind_speed:
        parser.error(
            f"--ground-speed ({args.ground_speed} m/s) must be greater than "
            f"--wind-speed ({args.wind_speed} m/s)."
        )

    try:
        stats = plan(
            kml_path=args.kml,
            wind_dir_from_deg=args.wind_dir,
            wind_speed_ms=args.wind_speed,
            altitude_m=args.altitude,
            cell_w=cam.cell_w,
            cell_h=cam.cell_h,
            ground_speed_ms=args.ground_speed,
            trigger_distance_m=trigger,
            overlap_side=args.overlap_side,
            overlap_front=args.overlap_front,
            optimize_speed=not args.no_optimize_speed,
            min_groundspeed_ms=args.min_groundspeed,
            max_groundspeed_ms=args.max_groundspeed,
            transition_speed_ms=args.transition_speed,
            camera_fps=args.camera_fps,
            output_kml=out_kml,
            output_wp=out_wp,
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print("\nSummary:")
        print(f"  Cells retained : {stats['n_cells']}")
        print(f"  Strips         : {stats['n_strips']}")
        print(f"  Waypoints      : {stats['n_waypoints']}")
        print(f"  Estimated time : {stats['total_time_s']:.0f} s "
              f"({stats['total_time_s']/60:.1f} min)")
        print(f"  Path length    : {stats['total_length_m']/1000:.2f} km")
        print(f"  Turns          : {stats['turn_count']}")
        print(f"  Mean strip spd : {stats['mean_strip_speed_ms']:.1f} m/s")
        print(f"\nOutputs written:")
        print(f"  Google Earth   : {out_kml}")
        print(f"  Mission Planner: {out_wp}")


if __name__ == "__main__":
    main()
