"""
fig_farmland_paths.py
=====================
Generates paper/figures/fig_farmland_paths.pdf

2×3 grid showing all six planning configurations on the SE Michigan
farmland polygon (Farmland.kml):

  R1  Low altitude (50 m, 70/80 %)   |  R2  Baseline (80 m, 70/80 %)
  R3  High altitude (120 m, 70/80 %)  |  R4  Dense (80 m, 80/85 %)
  R5  Rapid (80 m, 60/70 %)           |  R6  Wind 180 deg / 6 m/s

Each panel shows the continuous flight path with:
  - Survey segments: solid coloured line
  - Inter-strip transitions: dashed grey line
  - Periodic arrowheads along survey segments indicating flight direction
  - Survey polygon (transparent fill, solid boundary)

Usage
-----
    python code/fig_farmland_paths.py
"""

from __future__ import annotations

import csv
import sys
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import colors as mcolors
from matplotlib.patches import Polygon as MplPoly, FancyArrowPatch
from matplotlib.collections import LineCollection
from shapely.geometry import Polygon

# Add code/ directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "gcs_plugin"))

from decomposition import decompose
from routing import (
    build_route,
    optimal_grid_angle_energy,
    DEFAULT_POWER_MODEL,
)
from gcs_plugin.coords import parse_kml_polygon, lonlat_to_utm_polygon
from gcs_plugin.camera import camera_from_preset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

KML = Path(__file__).parent.parent / "code" / "scenarios" / "Farmland.kml"
OUT = Path(__file__).parent.parent / "paper" / "figures" / "fig_farmland_paths.pdf"
SUMMARY_CSV = Path(__file__).parent.parent / "paper" / "data" / "farmland_paths" / "farmland_runs_summary.csv"

# Runs: (run_id, title, alt, overlap_side, overlap_front,
#        wind_dir_from_deg, wind_speed_ms, nominal_speed_ms)
RUNS = [
    ("R1", "Low alt. (50 m)",
     50,  0.70, 0.80, 45, 3.0, 8.0),
    ("R2", "Baseline (80 m)",
     80,  0.70, 0.80, 45, 3.0, 8.0),
    ("R3", "High alt. (120 m)",
     120, 0.70, 0.80, 45, 3.0, 8.0),
    ("R4", "Dense (80 m)",
     80,  0.80, 0.85, 45, 3.0, 8.0),
    ("R5", "Rapid (80 m)",
     80,  0.60, 0.70, 45, 3.0, 8.0),
    ("R6", r"Wind $180^\circ$, 6 m/s",
     80,  0.70, 0.80, 180, 6.0, 10.0),
]

# Colours
C_SURVEY    = "#e41a1c"      # solid red for survey segments
C_TRANSIT   = "#aaaaaa"      # dashed grey for inter-strip transit
C_POLY_EDGE = "#2c7bb6"
C_POLY_FACE = "#abd9e9"

# Arrowhead spacing: place an arrow roughly every ARROW_SPACING metres
ARROW_SPACING_M = 80.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def met_to_wind_vec(wind_dir_from_deg: float, wind_speed_ms: float) -> np.ndarray:
    toward_deg = (wind_dir_from_deg + 180.0) % 360.0
    rad = math.radians(90.0 - toward_deg)
    return np.array([math.cos(rad), math.sin(rad)]) * wind_speed_ms

def wind_grid_angle(wind_dir_from_deg: float) -> float:
    toward_deg = (wind_dir_from_deg + 180.0) % 360.0
    return (90.0 - toward_deg) % 180.0


def get_route(kml: Path, alt: float, os: float, of: float,
              wd: float, ws: float, gs: float):
    """Return route artifacts and metrics for one farmland run."""
    lonlat_verts = parse_kml_polygon(str(kml))
    polygon_utm, crs = lonlat_to_utm_polygon(lonlat_verts)
    cp = camera_from_preset("sony_a6000_16mm", alt, os, of)
    wind_vec = met_to_wind_vec(wd, ws)
    phi_deg = optimal_grid_angle_energy(
        polygon_utm,
        cp.cell_w,
        cp.cell_h,
        wind_vec,
        gs,
        power_model=DEFAULT_POWER_MODEL,
    )
    decomp = decompose(polygon_utm, cp.cell_w, cp.cell_h, angle_deg=phi_deg)
    route = build_route(
        decomp,
        wind_vec,
        airspeed=gs,
        power_model=DEFAULT_POWER_MODEL,
        objective="energy",
    )

    poly_coords = np.array(polygon_utm.exterior.coords)
    cx, cy = poly_coords[:, 0].mean(), poly_coords[:, 1].mean()

    poly_local = poly_coords - [cx, cy]
    wps_local  = np.array([[w[0] - cx, w[1] - cy] for w in route.waypoints])
    t_set      = set(route.transition_indices)   # indices of transit-entry waypoints

    return {
        "poly_local": poly_local,
        "wps_local": wps_local,
        "t_set": t_set,
        "phi_deg": float(phi_deg),
        "n_cells": int(decomp.n_cells),
        "n_strips": int(route.metrics["n_strips"]),
        "n_waypoints": int(len(route.waypoints)),
        "path_km": float(route.total_length_m / 1000.0),
        "est_time_min": float(route.total_time_s / 60.0),
        "energy_kJ": float(route.total_energy_J / 1000.0),
        "mean_airspeed_mps": float(route.metrics.get("mean_airspeed_mps", gs)),
        "leg_ground_speeds_mps": [float(v) for v in route.leg_ground_speeds_mps],
        "cell_w_m": float(cp.cell_w),
        "cell_h_m": float(cp.cell_h),
        "gsd_cm": float(getattr(cp, "gsd_cm_per_px", 0.0)),
    }


def add_arrows(ax, pts: np.ndarray, spacing_m: float,
               color: str, arrowsize: float = 6.0):
    """
    Place small filled arrowheads along a polyline at roughly `spacing_m` intervals.
    `pts` is (N, 2) array of x, y coordinates.
    """
    dist = 0.0
    next_arrow = spacing_m * 0.5   # start mid-segment so arrows don't pile up at start

    for i in range(len(pts) - 1):
        seg = pts[i + 1] - pts[i]
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-6:
            continue

        # Walk along this segment placing arrows
        seg_walked = 0.0
        while seg_walked + (next_arrow - dist) <= seg_len:
            seg_walked += (next_arrow - dist)
            dist = 0.0
            next_arrow = spacing_m

            # Position and direction of arrow
            frac = seg_walked / seg_len
            px, py = pts[i] + frac * seg
            dx, dy = seg / seg_len

            # Draw a small filled triangle arrow using annotate
            ax.annotate("",
                xy=(px + dx * 0.1, py + dy * 0.1),
                xytext=(px - dx * 0.1, py - dy * 0.1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=0.0,
                    mutation_scale=arrowsize,
                ),
                zorder=6,
            )

        dist += seg_len - seg_walked


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    2, 3,
    figsize=(7.0, 4.8),
    constrained_layout=True,
)

# Shared axis range from baseline (all runs have same polygon)
shared = get_route(KML, 80, 0.70, 0.80, 45, 3.0, 8.0)
poly_shared = shared["poly_local"]
_xspan = np.ptp(poly_shared[:, 0])
_yspan = np.ptp(poly_shared[:, 1])
_ext   = max(_xspan, _yspan)
_pad   = 0.12 * _ext
_cx    = poly_shared[:, 0].mean()
_cy    = poly_shared[:, 1].mean()
XLIM = (_cx - _ext / 2 - _pad, _cx + _ext / 2 + _pad)
YLIM = (_cy - _ext / 2 - _pad, _cy + _ext / 2 + _pad)

summary_rows = []
run_outputs = [get_route(KML, alt, os, of, wd, ws, gs) for (_, _, alt, os, of, wd, ws, gs) in RUNS]
all_vg = [v for out in run_outputs for v in out["leg_ground_speeds_mps"]]
vg_min = min(all_vg) if all_vg else 0.0
vg_max = max(all_vg) if all_vg else 1.0
norm = mcolors.Normalize(vmin=vg_min, vmax=vg_max)
cmap = plt.get_cmap("turbo")

for ax, (run_spec, out) in zip(axes.flat, zip(RUNS, run_outputs)):

    run_id, run_title, alt, os, of, wd, ws, gs = run_spec

    poly_xy = out["poly_local"]
    wps_xy = out["wps_local"]
    t_set = out["t_set"]
    leg_vg = out["leg_ground_speeds_mps"]

    summary_rows.append({
        "id": run_id,
        "tag": f"farmland_{run_id.lower()}",
        "objective": "energy",
        "altitude_m": alt,
        "overlap_side_pct": int(round(os * 100)),
        "overlap_front_pct": int(round(of * 100)),
        "wind_dir_deg": wd,
        "wind_speed_ms": ws,
        "nominal_speed_ms": gs,
        "cell_w_m": out["cell_w_m"],
        "cell_h_m": out["cell_h_m"],
        "gsd_cm": out["gsd_cm"],
        "n_cells": out["n_cells"],
        "n_strips": out["n_strips"],
        "n_waypoints": out["n_waypoints"],
        "path_km": round(out["path_km"], 2),
        "est_time_min": round(out["est_time_min"], 2),
        "energy_kJ": round(out["energy_kJ"], 2),
        "mean_airspeed_mps": round(out["mean_airspeed_mps"], 2),
        "phi_opt_deg": round(out["phi_deg"], 1),
    })

    # ── Polygon ──────────────────────────────────────────────────────────
    poly_patch = MplPoly(poly_xy, closed=True,
                         facecolor=C_POLY_FACE, edgecolor=C_POLY_EDGE,
                         linewidth=1.0, alpha=0.45, zorder=2)
    ax.add_patch(poly_patch)

    # ── Flight path: split into survey and transit segments ───────────────
    # A segment from wps_xy[k] to wps_xy[k+1] is a TRANSIT segment if
    # index k+1 is in t_set (marks the start of the next strip after transit).
    survey_segs = []
    survey_vg = []
    transit_segs = []
    # Keep each continuous strip run so we can color it with one representative speed.
    survey_runs_pts: list[list[np.ndarray]] = []
    survey_runs_segidx: list[list[int]] = []
    in_survey_run_pts: list[np.ndarray] = []
    in_survey_run_segidx: list[int] = []

    # leg_vg indexes only non-zero legs. Waypoint polylines can contain
    # zero-length duplicates at strip boundaries, so track leg index separately.
    leg_idx = 0
    for k in range(len(wps_xy) - 1):
        p0, p1 = wps_xy[k], wps_xy[k + 1]
        seg_len = float(np.linalg.norm(p1 - p0))
        if seg_len < 1e-9:
            continue

        vg_k = leg_vg[leg_idx] if leg_idx < len(leg_vg) else np.nan
        leg_idx += 1

        if (k + 1) in t_set:
            # This segment is a transit (exit → entry of next strip)
            transit_segs.append([p0, p1])
            if in_survey_run_pts:
                survey_runs_pts.append(in_survey_run_pts)
                survey_runs_segidx.append(in_survey_run_segidx)
                in_survey_run_pts = []
                in_survey_run_segidx = []
        else:
            survey_segs.append([p0, p1])
            survey_vg.append(vg_k)
            survey_seg_idx = len(survey_segs) - 1
            if not in_survey_run_pts:
                in_survey_run_pts = [p0]
            in_survey_run_pts.append(p1)
            in_survey_run_segidx.append(survey_seg_idx)

    if in_survey_run_pts:
        survey_runs_pts.append(in_survey_run_pts)
        survey_runs_segidx.append(in_survey_run_segidx)

    # Draw transit segments: dashed grey
    if transit_segs:
        lc_t = LineCollection(transit_segs, colors=C_TRANSIT,
                              linewidths=0.6, linestyles="--", zorder=3)
        ax.add_collection(lc_t)

    # Draw survey segments color-coded by ground speed.
    # Use one speed per continuous strip run to avoid false intra-strip color jumps.
    if survey_segs:
        run_means = []
        for idxs in survey_runs_segidx:
            vals = [survey_vg[i] for i in idxs if np.isfinite(survey_vg[i])]
            run_means.append(float(np.mean(vals)) if vals else np.nan)
        for idxs, run_vg in zip(survey_runs_segidx, run_means):
            if not idxs:
                continue
            segs = [survey_segs[i] for i in idxs]
            color = cmap(norm(run_vg)) if np.isfinite(run_vg) else "#888888"
            lc_run = LineCollection(segs, colors=[color], linewidths=0.9, zorder=4)
            ax.add_collection(lc_run)

    # ── Arrowheads along survey segments only ─────────────────────────────
    arrow_spacing = ARROW_SPACING_M * (alt / 80.0)   # scale with cell density
    for run in survey_runs_pts:
        pts = np.array(run)
        add_arrows(ax, pts, spacing_m=arrow_spacing,
                   color="#404040", arrowsize=5.0)

    # ── Start / end markers ───────────────────────────────────────────────
    ax.plot(*wps_xy[0],  "^", ms=3.5, color="#2ca02c", zorder=7, clip_on=False)
    ax.plot(*wps_xy[-1], "s", ms=3.0, color="#555555", zorder=7, clip_on=False)

    # ── Axes cosmetics ────────────────────────────────────────────────────
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal")
    ax.axis("off")

    # Scale bar (100 m)
    sb_x = XLIM[0] + 0.05 * (XLIM[1] - XLIM[0])
    sb_y = YLIM[0] + 0.07 * (YLIM[1] - YLIM[0])
    ax.plot([sb_x, sb_x + 100], [sb_y, sb_y], "-", color="k", lw=1.5, zorder=8)
    ax.text(sb_x + 50, sb_y + 8, "100 m",
            ha="center", va="bottom", fontsize=9, color="k", zorder=8)

    # Run ID (top-left, bold)
    ax.text(XLIM[0] + 0.04 * (XLIM[1] - XLIM[0]),
            YLIM[1] - 0.04 * (YLIM[1] - YLIM[0]),
            run_id, fontsize=9, fontweight="bold", va="top",
            color="k", zorder=9)

    # Stats annotation (bottom-right)
    stats_text = "\n".join([
        f"{int(round(os*100))}/{int(round(of*100))}%,  $\\varphi^*={out['phi_deg']:.0f}^\\circ$",
        rf"{out['n_strips']} strips, {out['path_km']:.1f} km",
        rf"$E={out['energy_kJ']:.1f}$ kJ, $T={out['est_time_min']:.1f}$ min",
    ])
    ax.text(XLIM[1] - 0.04 * (XLIM[1] - XLIM[0]),
            YLIM[0] + 0.04 * (YLIM[1] - YLIM[0]),
            stats_text, fontsize=8, va="bottom", ha="right",
            color="#222222", zorder=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))


# ── Shared legend ──────────────────────────────────────────────────────────
h_survey  = mlines.Line2D([], [], color="#666666", lw=1.2,
                           label="Survey segment")
h_transit = mlines.Line2D([], [], color=C_TRANSIT, lw=1.0,
                           linestyle="--", label="Inter-strip transit")
h_start   = mlines.Line2D([], [], marker="^", color="#2ca02c",
                           ms=4, linestyle="none", label="Start")
h_end     = mlines.Line2D([], [], marker="s", color="#555555",
                           ms=3.5, linestyle="none", label="End")
fig.legend(handles=[h_survey, h_transit, h_start, h_end],
           loc="lower center", ncol=4, fontsize=8,
           frameon=True, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.02))

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.022, pad=0.01)
cbar.set_label("Leg ground speed (m/s)", fontsize=11)
cbar.ax.tick_params(labelsize=9)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved to {OUT}")

SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
with SUMMARY_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"Saved to {SUMMARY_CSV}")
