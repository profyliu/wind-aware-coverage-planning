"""
visualize.py
============
Graphical illustration of polygon decomposition and coverage paths.

Produces:
  - fig_coverage_paths.pdf  : panel (a)+(b)+(c) comparing proposed method
                               and two baselines
  - fig_wind_sweep.pdf      : estimated flight time vs wind direction
  - fig_cell_count.pdf      : retained-cell count vs concavity / cell size
  - fig_runtime.pdf         : planner runtime vs instance size

Usage
-----
    python visualize.py                  # generates all stock figures
    python visualize.py --scenario 01    # generates only scenario-01 panel
"""

import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon

# Ensure local modules are importable
sys.path.insert(0, str(Path(__file__).parent))

FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLOUR = {
    "polygon":    "#2c7bb6",
    "interior":   "#abd9e9",
    "boundary":   "#fdae61",
    "path":       "#d7191c",
    "transition": "#756bb1",
    "wind":       "#1a9641",
    "baseline1":  "#636363",
    "baseline2":  "#bdbdbd",
}


# ---------------------------------------------------------------------------
# Core drawing functions
# ---------------------------------------------------------------------------

def draw_polygon(ax: plt.Axes, poly: Polygon,
                 facecolor: str = COLOUR["polygon"],
                 alpha: float = 0.15,
                 edgecolor: str = COLOUR["polygon"],
                 linewidth: float = 1.5,
                 label: str = "Region P") -> None:
    coords = np.array(poly.exterior.coords)
    # Faint fill in the background
    patch = MplPolygon(coords, closed=True,
                        facecolor=facecolor, edgecolor="none",
                        linewidth=0, alpha=alpha, label=label, zorder=1)
    ax.add_patch(patch)
    # High-contrast outline drawn in the foreground, above cells and paths
    outline = MplPolygon(coords, closed=True,
                          facecolor="none", edgecolor="#111111",
                          linewidth=1.2, alpha=1.0, zorder=10)
    ax.add_patch(outline)


def draw_cells(ax: plt.Axes, cells, show_boundary: bool = True) -> None:
    """Draw all retained cells colour-coded by type."""
    from decomposition import Cell
    for cell in cells:
        coords = np.array(cell.footprint.exterior.coords)
        fc = COLOUR["boundary"] if cell.is_boundary else COLOUR["interior"]
        patch = MplPolygon(coords, closed=True,
                            facecolor=fc, edgecolor="k",
                            linewidth=0.4, alpha=0.6, zorder=2)
        ax.add_patch(patch)


def draw_path(ax: plt.Axes, waypoints: List[np.ndarray],
              transition_indices: Optional[List[int]] = None,
              color: str = COLOUR["path"],
              linewidth: float = 1.2,
              label: str = "Coverage path",
              alpha: float = 0.85) -> None:
    """Draw coverage path as a polyline connecting successive cell centres.

    Parameters
    ----------
    waypoints          : ordered list of 2-D position arrays (one per cell centre
                         plus inter-strip transition points).
    transition_indices : optional list of waypoint indices that are inter-strip
                         transition start points.  If provided, segments leading
                         into those indices are drawn in a distinct dashed style
                         to make inter-strip hops visually distinguishable from
                         within-strip coverage segments.
    """
    if len(waypoints) < 2:
        return

    trans_set = set(transition_indices) if transition_indices else set()

    # Build segment lists: within-strip (solid red) vs transition (dashed purple)
    coverage_xs, coverage_ys = [], []
    trans_xs, trans_ys = [], []

    for i in range(len(waypoints) - 1):
        p1, p2 = waypoints[i], waypoints[i + 1]
        # A transition segment ends at waypoint i+1 if i+1 is a transition index
        if (i + 1) in trans_set:
            trans_xs += [p1[0], p2[0], None]
            trans_ys += [p1[1], p2[1], None]
        else:
            coverage_xs += [p1[0], p2[0], None]
            coverage_ys += [p1[1], p2[1], None]

    # Draw within-strip segments (solid red)
    if coverage_xs:
        ax.plot(coverage_xs, coverage_ys, color=color, linewidth=linewidth,
                label=label, alpha=alpha, zorder=4, solid_capstyle="round")

    # Draw inter-strip transition segments (dashed purple)
    if trans_xs:
        ax.plot(trans_xs, trans_ys,
                color=COLOUR["transition"], linewidth=max(linewidth * 0.8, 0.9),
                linestyle="--", alpha=alpha * 0.85, zorder=4,
                label="Inter-strip transition")

    # Add direction arrows every ~10% of waypoints (at least 1, at most 5)
    n = len(waypoints)
    step = max(1, n // 8)
    for i in range(0, n - 1, step):
        p1, p2 = waypoints[i], waypoints[i + 1]
        mid = (p1 + p2) / 2.0
        dv = p2 - p1
        d = np.linalg.norm(dv)
        if d > 1e-6:
            ax.annotate("", xy=mid + dv / d * 3, xytext=mid,
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=0.8, mutation_scale=6),
                        zorder=5)

    # Start / end markers
    xs = [p[0] for p in waypoints]
    ys = [p[1] for p in waypoints]
    ax.plot(xs[0], ys[0], "go", markersize=6, zorder=5, label="Start")
    ax.plot(xs[-1], ys[-1], "rs", markersize=6, zorder=5, label="End")


def draw_wind_arrow(ax: plt.Axes, wind_vec: np.ndarray, origin: np.ndarray,
                    scale: float = 60.0) -> None:
    """Draw a single arrow indicating wind direction and magnitude."""
    ax.annotate(
        "", xy=origin + wind_vec / np.linalg.norm(wind_vec) * scale,
        xytext=origin,
        arrowprops=dict(arrowstyle="->", color=COLOUR["wind"], lw=2),
        zorder=6,
    )
    ax.text(*(origin + wind_vec / np.linalg.norm(wind_vec) * scale * 1.15),
             f"Wind\n{np.linalg.norm(wind_vec):.1f} m/s",
             ha="center", va="center", fontsize=9, color=COLOUR["wind"])


def _axes_bounds(poly: Polygon, margin_frac: float = 0.05):
    minx, miny, maxx, maxy = poly.bounds
    dx = (maxx - minx) * margin_frac
    dy = (maxy - miny) * margin_frac
    return minx - dx, maxx + dx, miny - dy, maxy + dy


# ---------------------------------------------------------------------------
# Figure 1 – Coverage-path panel (three sub-panels)
# ---------------------------------------------------------------------------

def figure_coverage_paths(polygon: Polygon,
                           proposed_decomp, proposed_route,
                           baseline1_decomp, baseline1_route,
                           baseline2_decomp, baseline2_route,
                           wind_vec: np.ndarray,
                           out_path: Optional[Path] = None,
                           baseline3_decomp=None, baseline3_route=None) -> Path:
    """
    2×2 panel figure:
      (a) Proposed method (energy-optimal orientation, energy-optimal speed)
      (b) Baseline 1 (fixed 0° orientation, fixed speed)
      (c) Baseline 2 (time-optimal orientation, fixed speed)
      (d) Baseline 3 (fixed 0° orientation, energy-optimal speed)
    If baseline3_decomp/baseline3_route are None, falls back to a 1×3 layout.
    """
    if baseline3_decomp is not None and baseline3_route is not None:
        fig, axes_grid = plt.subplots(2, 2, figsize=(9.0, 7.5))
        axes = axes_grid.flatten()
        titles = [
            "(a) Proposed: energy-optimal $\\phi^*$, energy-optimal speed",
            "(b) Baseline 1: $\\phi=0°$, fixed speed",
            "(c) Baseline 2: time-optimal $\\phi^*$, fixed speed",
            "(d) Baseline 3: $\\phi=0°$, energy-optimal speed",
        ]
        configs = [
            (proposed_decomp,  proposed_route),
            (baseline1_decomp, baseline1_route),
            (baseline2_decomp, baseline2_route),
            (baseline3_decomp, baseline3_route),
        ]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.5))
        titles = [
            "(a) Proposed: energy-optimal selective grid",
            "(b) Baseline 1: fixed-orientation selective grid",
            "(c) Baseline 2: time-optimal orientation",
        ]
        configs = [
            (proposed_decomp,  proposed_route),
            (baseline1_decomp, baseline1_route),
            (baseline2_decomp, baseline2_route),
        ]

    arrow_origin = None
    for ax, title, (decomp, route) in zip(axes, titles, configs):
        draw_polygon(ax, polygon)
        draw_cells(ax, decomp.cells)
        draw_path(ax, route.waypoints,
                  transition_indices=getattr(route, "transition_indices", None))

        # Wind arrow (once, based on polygon bounds)
        if arrow_origin is None:
            minx, miny, maxx, maxy = polygon.bounds
            arrow_origin = np.array([minx + (maxx - minx) * 0.1,
                                      maxy - (maxy - miny) * 0.15])
        draw_wind_arrow(ax, wind_vec, arrow_origin)

        xlim = _axes_bounds(polygon)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(xlim[2], xlim[3])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("x (m)", fontsize=13)
        ax.set_ylabel("y (m)", fontsize=13)
        ax.tick_params(labelsize=11)

        # Stats annotation
        if getattr(route, "total_energy_J", 0.0) > 0:
            stats = (f"n={decomp.n_cells} cells\n"
                     f"E={route.total_energy_J/1000.0:.1f} kJ\n"
                     f"t={route.total_time_s:.0f} s\n"
                     f"L={route.total_length_m:.0f} m")
        else:
            stats = (f"n={decomp.n_cells} cells\n"
                     f"t={route.total_time_s:.0f} s\n"
                     f"L={route.total_length_m:.0f} m")
        ax.text(0.98, 0.02, stats, transform=ax.transAxes,
                 fontsize=9, ha="right", va="bottom", zorder=20,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#cccccc", lw=0.5))

        # Legend (first panel only to avoid clutter)
        if ax is axes[0]:
            from matplotlib.lines import Line2D
            legend_handles = [
                mpatches.Patch(fc=COLOUR["interior"], ec="k", lw=0.4, label="Interior cell"),
                mpatches.Patch(fc=COLOUR["boundary"], ec="k", lw=0.4, label="Boundary cell"),
                mpatches.Patch(fc=COLOUR["polygon"], alpha=0.3, ec=COLOUR["polygon"], label="Region P"),
                Line2D([0], [0], color=COLOUR["path"], lw=1.4, label="Coverage path"),
                Line2D([0], [0], color=COLOUR["transition"], lw=1.2,
                       linestyle="--", label="Transition"),
            ]
            ax.legend(handles=legend_handles, fontsize=9, loc="upper right")

    plt.tight_layout(pad=0.8)
    if out_path is None:
        out_path = FIGURES_DIR / "fig_coverage_paths.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 2 – Flight time vs wind direction
# ---------------------------------------------------------------------------

def figure_wind_sweep(wind_dirs_deg: List[float],
                       proposed_energy: List[float],
                       baseline_energy: List[float],
                       nowind_energy: float,
                       out_path: Optional[Path] = None) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    ax.plot(wind_dirs_deg, proposed_energy, "o-",
             color=COLOUR["path"], linewidth=1.5, markersize=5,
             label="Proposed (energy-optimal)")
    ax.plot(wind_dirs_deg, baseline_energy, "s--",
             color=COLOUR["baseline1"], linewidth=1.5, markersize=5,
             label="Baseline (fixed orientation)")
    ax.axhline(nowind_energy, color=COLOUR["polygon"], linestyle=":",
                linewidth=1.2, label="No-wind reference")

    ax.set_xlabel("Wind FROM bearing (°)", fontsize=13)
    ax.set_ylabel("Estimated mission energy (kJ)", fontsize=13)
    ax.set_title("Estimated mission energy vs. wind source direction\n(Concave Farmland scenario)",
                  fontsize=13)
    ax.set_xticks(wind_dirs_deg)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(pad=0.8)

    if out_path is None:
        out_path = FIGURES_DIR / "fig_wind_sweep.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 3 – Retained-cell count vs concavity / cell size
# ---------------------------------------------------------------------------

def figure_cell_count(results_df,
                       out_path: Optional[Path] = None) -> Path:
    """
    results_df must have columns: family, cell_size, n_cells
    Left panel: log-log plot of n vs s with O(1/s²) reference line.
    Right panel: box-plot of n by family (all cell sizes pooled by family+size).
    """
    import pandas as pd

    FAM_LABELS = {
        "convex":       "Convex",
        "mild_concave": "Mild concave",
        "high_concave": "High concave",
        "corridor":     "Corridor",
    }
    PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    families = sorted(results_df["family"].unique(),
                      key=lambda f: list(FAM_LABELS.keys()).index(f)
                                    if f in FAM_LABELS else 99)
    cell_sizes = sorted(results_df["cell_size"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # ------------------------------------------------------------------
    # Left: log-log n vs s per family + O(1/s²) reference line
    # ------------------------------------------------------------------
    ax = axes[0]
    for fam, colour in zip(families, PALETTE):
        sub = results_df[results_df["family"] == fam].sort_values("cell_size")
        label = FAM_LABELS.get(fam, fam)
        ax.plot(sub["cell_size"], sub["n_cells"], "o-",
                color=colour, markersize=6, linewidth=1.6, label=label, zorder=3)

    # Reference O(1/s²) line anchored at the smallest cell size of the first family
    anchor_fam = families[0]
    anchor_sub = results_df[results_df["family"] == anchor_fam].sort_values("cell_size")
    s0 = anchor_sub["cell_size"].iloc[0]
    n0 = anchor_sub["n_cells"].iloc[0]
    s_ref = np.array(cell_sizes, dtype=float)
    n_ref = n0 * (s0 / s_ref) ** 2
    ax.plot(s_ref, n_ref, "k--", linewidth=1.0, alpha=0.6, label=r"$\propto 1/s^2$", zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(cell_sizes)
    ax.set_xticklabels([str(int(s)) for s in cell_sizes], fontsize=11)
    ax.set_xlabel("Cell size $s$ (m)", fontsize=13)
    ax.set_ylabel("Mean retained cells $n$", fontsize=13)
    ax.set_title(r"Retained cells vs.\ cell size (log--log)", fontsize=13)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)

    # ------------------------------------------------------------------
    # Right: grouped bar chart — mean n by family for each cell size
    # ------------------------------------------------------------------
    ax = axes[1]
    n_fam = len(families)
    n_sizes = len(cell_sizes)
    bar_w = 0.7 / n_fam
    x = np.arange(n_sizes)

    for i, (fam, colour) in enumerate(zip(families, PALETTE)):
        means = [results_df[(results_df["family"] == fam) &
                            (results_df["cell_size"] == s)]["n_cells"].mean()
                 for s in cell_sizes]
        offset = (i - (n_fam - 1) / 2) * bar_w
        ax.bar(x + offset, means, bar_w * 0.92,
               color=colour, label=FAM_LABELS.get(fam, fam), alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels([f"s={int(s)} m" for s in cell_sizes], fontsize=11)
    ax.set_xlabel("Cell size $s$ (m)", fontsize=13)
    ax.set_ylabel("Mean retained cells $n$", fontsize=13)
    ax.set_title("Retained cells by family and cell size", fontsize=13)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout(pad=0.8)
    if out_path is None:
        out_path = FIGURES_DIR / "fig_cell_count.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 4 – Planner runtime vs instance size
# ---------------------------------------------------------------------------

def figure_runtime(results_df,
                    out_path: Optional[Path] = None) -> Path:
    """
    results_df must have columns: n_cells, runtime_s
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.scatter(results_df["n_cells"], results_df["runtime_s"],
                alpha=0.6, s=20, color=COLOUR["polygon"])
    ax.set_xlabel("Retained cells n", fontsize=13)
    ax.set_ylabel("Planner runtime (s)", fontsize=13)
    ax.set_title("Planner runtime vs. problem size", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout(pad=0.8)
    if out_path is None:
        out_path = FIGURES_DIR / "fig_runtime.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Single polygon + path quick-view (utility)
# ---------------------------------------------------------------------------

def quick_plot(polygon: Polygon, decomp, route,
               wind_vec: np.ndarray,
               title: str = "Coverage path",
               out_path: Optional[Path] = None) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_polygon(ax, polygon)
    draw_cells(ax, decomp.cells)
    draw_path(ax, route.waypoints,
              transition_indices=getattr(route, "transition_indices", None))
    minx, miny, maxx, maxy = polygon.bounds
    arrow_origin = np.array([minx + (maxx - minx) * 0.08,
                              maxy - (maxy - miny) * 0.12])
    draw_wind_arrow(ax, wind_vec, arrow_origin)
    xlim = _axes_bounds(polygon)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(xlim[2], xlim[3])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    plt.tight_layout()
    if out_path is None:
        out_path = FIGURES_DIR / "quick_plot.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 5 – Polygon family shape illustrations
# ---------------------------------------------------------------------------

def figure_polygon_families(out_path: Optional[Path] = None) -> Path:
    """
    2×2 panel showing one representative polygon from each of the four
    benchmark families.  No grid or path overlay — just the polygon shape.
    Illustrates the geometric diversity tested in the experiments.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from instance_gen import generate_convex, generate_corridor
    from shapely.affinity import rotate as shp_rotate, scale as shp_scale
    from shapely.geometry import Polygon as ShapelyPolygon
    import math

    # ── Panel (b): actual Concave Farmland polygon from scenario_01.json ──────
    _farm_coords = [
        (0.0, 0.0), (120.0, -20.0), (240.0, 10.0), (360.0, -10.0), (480.0, 0.0),
        (500.0, 150.0), (480.0, 300.0), (400.0, 400.0), (250.0, 420.0),
        (100.0, 400.0), (20.0, 300.0), (0.0, 150.0),
    ]
    _farm_raw = ShapelyPolygon(_farm_coords)
    _fmn, _fmb, _fmx, _fmy = _farm_raw.bounds
    _fdiag = ((_fmx - _fmn) ** 2 + (_fmy - _fmb) ** 2) ** 0.5
    _farm_poly = shp_scale(_farm_raw, xfact=500.0 / _fdiag, yfact=500.0 / _fdiag,
                            origin=(_fmn, _fmb))

    # ── Panel (c): irregular 5-armed star polygon (high concave) ─────────────
    # 5 spiky arms of different lengths, deep concave bays between them
    _outer_angles = [90, 162, 234, 306, 18]   # tip angles (deg CCW from +x)
    _outer_radii  = [220, 195, 215, 190, 210]  # arm lengths (irregular)
    _inner_r      = 80                          # depth of concave bay
    _star_coords  = []
    for i in range(5):
        theta_o = math.radians(_outer_angles[i])
        _star_coords.append((_outer_radii[i] * math.cos(theta_o),
                              _outer_radii[i] * math.sin(theta_o)))
        theta_i = math.radians(_outer_angles[i] + 36)   # midpoint between tips
        _star_coords.append((_inner_r * math.cos(theta_i),
                              _inner_r * math.sin(theta_i)))
    _star_raw = ShapelyPolygon(_star_coords)
    _smn, _smb, _smx, _smy = _star_raw.bounds
    _sdiag = ((_smx - _smn) ** 2 + (_smy - _smb) ** 2) ** 0.5
    _star_poly = shp_scale(_star_raw, xfact=500.0 / _sdiag, yfact=500.0 / _sdiag,
                            origin=(_smn, _smb))

    specs = [
        ("(a) Convex",
         generate_convex(n_vertices=10, seed=1),
         "10-vertex convex field",
         "#2c7bb6"),
        ("(b) Mild concave",
         _farm_poly,
         "Concave Farmland scenario polygon\n(12-vertex, mildly concave)",
         "#1a9641"),
        ("(c) High concave",
         _star_poly,
         "5-armed irregular star\n(deep concave bays between arms)",
         "#d7191c"),
        ("(d) Corridor",
         generate_corridor(n_chambers=2, corridor_width=40.0, seed=0),
         "Two chambers, 40 m passage",
         "#756bb1"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
    axes = axes.flatten()

    for ax, (title, poly, desc, fc) in zip(axes, specs):
        coords = np.array(poly.exterior.coords)
        patch = plt.Polygon(coords, closed=True,
                             facecolor=fc, edgecolor=fc,
                             linewidth=1.5, alpha=0.25)
        ax.add_patch(patch)
        ax.plot(coords[:, 0], coords[:, 1], color=fc, linewidth=1.5)

        minx, miny, maxx, maxy = poly.bounds
        mx = (maxx - minx) * 0.07
        my = (maxy - miny) * 0.07
        ax.set_xlim(minx - mx, maxx + mx)
        ax.set_ylim(miny - my, maxy + my)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("x (m)", fontsize=13)
        ax.set_ylabel("y (m)", fontsize=13)
        ax.tick_params(labelsize=11)

        # Annotate with area only (compact, bottom-right to avoid covering polygon)
        area = poly.area
        ax.text(0.97, 0.03, f"Area\u2009=\u2009{area/1e4:.1f}\u2009ha",
                transform=ax.transAxes, fontsize=9,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))

    plt.suptitle("Benchmark polygon families\n(bounding-box diagonal normalised to 500 m)",
                 fontsize=13, y=1.01)
    plt.tight_layout(pad=0.8)
    if out_path is None:
        out_path = FIGURES_DIR / "fig_polygon_families.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test: generate scenario-01 quick plot
    import json
    sys.path.insert(0, str(Path(__file__).parent))
    from instance_gen import load_scenario_polygon
    from decomposition import decompose
    from routing import build_route

    scenario_path = Path(__file__).parent / "scenarios" / "scenario_01.json"
    with open(scenario_path) as f:
        cfg = json.load(f)

    poly = load_scenario_polygon(str(scenario_path))
    wind_deg = cfg["wind"]["direction_deg"]
    wind_speed = cfg["wind"]["speed_mps"]
    airspeed = cfg["uav"]["airspeed_mps"]
    cell_w = cfg["cell"]["width_m"]
    cell_h = cfg["cell"]["height_m"]

    wind_rad = math.radians(wind_deg)
    wind_vec = np.array([wind_speed * math.cos(wind_rad),
                          wind_speed * math.sin(wind_rad)])

    decomp = decompose(poly, cell_w=cell_w, cell_h=cell_h, angle_deg=wind_deg)
    route = build_route(decomp, wind_vec=wind_vec, airspeed=airspeed)

    out = quick_plot(poly, decomp, route, wind_vec,
                      title="Scenario 01 – Wind-aligned selective grid")
    print(f"Saved quick plot to {out}")
    print(f"  n_cells={decomp.n_cells}, flight_time={route.total_time_s:.0f}s, "
          f"path_length={route.total_length_m:.0f}m, turns={route.turn_count}")
