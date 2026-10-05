"""
fig_rotation_illustration.py
============================
Generates paper/figures/fig_rotation_illustration.pdf

Three-panel illustration of the generic grid construction at a given
orientation angle phi.  The same procedure is repeated for every candidate
phi in the orientation sweep; the wind plays no role here.

  Panel (a) – Original frame
      The polygon P in world coordinates.  A green arrow shows the candidate
      strip sweep direction (phi = 45 deg CCW from the +x axis).

  Panel (b) – Rotated frame  (after R_{-phi})
      P rotated by -phi so that the candidate grid becomes axis-aligned.
      An axis-aligned uniform grid (grey lines) is overlaid on the rotated
      polygon.  Cells that pass the any-intersection test are highlighted
      (interior = blue, boundary = orange).

  Panel (c) – Back in original frame  (after R_{+phi})
      The retained cells rotated back by +phi, overlaid on P.
      The tilted grid at orientation phi is now visible.

Usage
-----
    python code/fig_rotation_illustration.py
"""

from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPoly, FancyArrowPatch
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, box
from shapely.affinity import rotate as shp_rotate

# ── Output path ──────────────────────────────────────────────────────────────
OUT = Path(__file__).parent.parent / "paper" / "figures" / "fig_rotation_illustration.pdf"

# ── Parameters ────────────────────────────────────────────────────────────────
PHI_DEG = 45.0          # grid orientation angle (candidate phi), degrees CCW from East
W, H    = 80.0, 80.0   # cell width × height (same units as polygon coords)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_POLY     = "#2c7bb6"
C_INTERIOR = "#abd9e9"
C_BOUNDARY = "#fdae61"
C_GRID     = "#888888"
C_WIND     = "#1a9641"
C_AXIS     = "#d7191c"
C_ARROW    = "#333333"
ALPHA_POLY = 0.30
ALPHA_CELL = 0.55

# ── Polygon definition ────────────────────────────────────────────────────────
# A mildly concave heptagon in world coordinates (units: metres)
RAW = np.array([
    (0,   0),
    (300, 0),
    (380, 120),
    (340, 280),
    (200, 340),
    (60,  300),
    (-40, 180),
])
# Shift centroid to origin for cleaner rotation centre
_c = RAW.mean(axis=0)
POLY_COORDS = RAW - _c          # ~[-220, 220] × [-170, 170]


# ── Helpers ───────────────────────────────────────────────────────────────────

def rot_mat(deg: float) -> np.ndarray:
    """2×2 CCW rotation matrix."""
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s], [s, c]])


def rotate_coords(pts: np.ndarray, deg: float) -> np.ndarray:
    return pts @ rot_mat(deg).T


def make_polygon(coords: np.ndarray) -> Polygon:
    return Polygon(coords)


def retained_cells(poly: Polygon, w: float, h: float):
    """
    Return (interior_cells, boundary_cells) as lists of shapely Polygons,
    each a w×h rectangle in the same frame as `poly`.
    """
    minx, miny, maxx, maxy = poly.bounds
    a_min = math.floor(minx / w) - 1
    a_max = math.ceil(maxx  / w) + 1
    b_min = math.floor(miny / h) - 1
    b_max = math.ceil(maxy  / h) + 1

    interior, boundary = [], []
    eps = 1e-9 * w * h
    for a in range(a_min, a_max + 1):
        for b in range(b_min, b_max + 1):
            cell = box(a * w, b * h, (a + 1) * w, (b + 1) * h)
            inter = cell.intersection(poly)
            if inter.is_empty:
                continue
            if inter.area < eps:
                continue
            if inter.area < w * h - eps:
                boundary.append(cell)
            else:
                interior.append(cell)
    return interior, boundary


def draw_poly_patch(ax, coords, fc, ec, alpha, lw=1.5, zorder=1, label=None):
    kw = dict(closed=True, facecolor=fc, edgecolor=ec,
              linewidth=lw, alpha=alpha, zorder=zorder)
    if label:
        kw["label"] = label
    ax.add_patch(MplPoly(coords, **kw))


def draw_grid_lines(ax, poly_rotated: Polygon, w: float, h: float,
                    color=C_GRID, lw=0.5, alpha=0.4):
    """Draw axis-aligned grid lines covering the bounding box of poly_rotated."""
    minx, miny, maxx, maxy = poly_rotated.bounds
    pad = max(w, h)
    x0 = math.floor((minx - pad) / w) * w
    x1 = math.ceil ((maxx + pad) / w) * w
    y0 = math.floor((miny - pad) / h) * h
    y1 = math.ceil ((maxy + pad) / h) * h
    xs = np.arange(x0, x1 + w, w)
    ys = np.arange(y0, y1 + h, h)
    for x in xs:
        ax.axvline(x, color=color, lw=lw, alpha=alpha, zorder=0)
    for y in ys:
        ax.axhline(y, color=color, lw=lw, alpha=alpha, zorder=0)


def sweep_arrow(ax, cx, cy, length, deg, color=C_WIND):
    """Draw a labelled strip-sweep-direction arrow at (cx, cy) at angle `deg` CCW from +x."""
    dx = length * math.cos(math.radians(deg))
    dy = length * math.sin(math.radians(deg))
    ax.annotate(
        "", xy=(cx + dx, cy + dy), xytext=(cx, cy),
        arrowprops=dict(arrowstyle="->, head_width=0.4, head_length=0.4",
                        color=color, lw=2.0),
        zorder=10,
    )
    ax.text(cx + dx * 1.65, cy + dy * 1.65,
            f"$\\hat{{e}}_{{\\phi}}$\n$\\phi={PHI_DEG:.0f}^{{\\circ}}$",
            ha="center", va="center", fontsize=10, color=color, zorder=10)


def phi_arc(ax, cx, cy, r, phi_deg, color=C_AXIS):
    """Draw an arc from 0 to phi_deg at radius r, centred at (cx,cy)."""
    theta = np.linspace(0, math.radians(phi_deg), 120)
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
            color=color, lw=1.2, zorder=8)
    # tick at 0 deg
    ax.plot([cx, cx + r], [cy, cy], color=color, lw=1.0, ls="--", zorder=7)
    # label
    mid = math.radians(phi_deg / 2)
    ax.text(cx + r * 1.35 * math.cos(mid), cy + r * 1.35 * math.sin(mid),
            f"$\\phi$", ha="center", va="center", fontsize=10, color=color)


def rotation_arc(ax, cx, cy, r, phi_deg, color="#e41a1c", label=""):
    """Curved arrow indicating a rotation of phi_deg."""
    from matplotlib.patches import Arc, FancyArrowPatch
    # draw arc
    theta_arr = np.linspace(math.radians(10), math.radians(phi_deg - 10), 80)
    ax.plot(cx + r * np.cos(theta_arr), cy + r * np.sin(theta_arr),
            color=color, lw=1.5, zorder=9)
    # arrowhead at end
    te = math.radians(phi_deg - 10)
    ax.annotate("", xy=(cx + r * math.cos(math.radians(phi_deg)),
                         cy + r * math.sin(math.radians(phi_deg))),
                xytext=(cx + r * math.cos(te), cy + r * math.sin(te)),
                arrowprops=dict(arrowstyle="->, head_width=0.3",
                                color=color, lw=1.5),
                zorder=9)
    if label:
        mid = math.radians(phi_deg * 0.5)
        ax.text(cx + r * 1.75 * math.cos(mid), cy + r * 1.75 * math.sin(mid),
                label, ha="center", va="center", fontsize=10,
                color=color, zorder=9)


def set_equal_ax(ax, data_pts, margin=40):
    xs, ys = data_pts[:, 0], data_pts[:, 1]
    cx, cy = xs.mean(), ys.mean()
    rng = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + margin
    ax.set_xlim(cx - rng, cx + rng)
    ax.set_ylim(cy - rng, cy + rng)
    ax.set_aspect("equal")


# ── Main figure ───────────────────────────────────────────────────────────────

def make_figure():
    phi = PHI_DEG

    # ── Data preparation ──────────────────────────────────────────────────────
    # (a) original-frame polygon
    poly_orig = make_polygon(POLY_COORDS)

    # (b) rotated polygon (rotate by -phi so candidate grid becomes axis-aligned)
    poly_rot_coords = rotate_coords(POLY_COORDS, -phi)
    poly_rot = make_polygon(poly_rot_coords)

    # Retained cells in the rotated frame
    int_cells_rot, bnd_cells_rot = retained_cells(poly_rot, W, H)

    # (c) rotate cells back to original frame
    def cell_coords_orig(shp_cell):
        c = np.array(shp_cell.exterior.coords[:-1])
        return rotate_coords(c, +phi)

    int_cells_orig = [cell_coords_orig(c) for c in int_cells_rot]
    bnd_cells_orig = [cell_coords_orig(c) for c in bnd_cells_rot]

    # ── Figure layout ─────────────────────────────────────────────────────────
    # figsize=7.0in → output 504pt; LaTeX textwidth 390pt → scale 0.774.
    # fontsize 13pt × 0.774 ≈ 10pt in the printed PDF.
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.8),
                             gridspec_kw={"wspace": 0.10})
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ════════════════════════════════════════════════════════════════
    # Panel (a) – original frame
    # ════════════════════════════════════════════════════════════════
    ax = axes[0]
    draw_poly_patch(ax, POLY_COORDS,
                    fc=C_POLY, ec=C_POLY, alpha=ALPHA_POLY, lw=1.8,
                    zorder=1, label="$P$")
    # Draw axis cross in original frame
    lim = 260
    ax.axhline(0, color="#aaa", lw=0.8, ls="--", zorder=0)
    ax.axvline(0, color="#aaa", lw=0.8, ls="--", zorder=0)
    ax.text( lim * 0.9,  8, "$x$", fontsize=10, color="#888")
    ax.text( 8,  lim * 0.9, "$y$", fontsize=10, color="#888")

    # Wind arrow  (direction = phi from the +x axis)
    arrow_len = 90
    ax_cx, ax_cy = -180, -130
    sweep_arrow(ax, ax_cx, ax_cy, arrow_len, phi)

    # phi arc showing the grid orientation
    phi_arc(ax, ax_cx, ax_cy, 55, phi)

    # Polygon label
    c = poly_orig.centroid
    ax.text(c.x, c.y + 20, "$P$", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_POLY)

    set_equal_ax(ax, POLY_COORDS, margin=80)
    ax.set_title("(a) Original frame", fontsize=10, pad=6)

    # ════════════════════════════════════════════════════════════════
    # Panel (b) – rotated frame
    # ════════════════════════════════════════════════════════════════
    ax = axes[1]

    # Grid lines in the rotated frame
    draw_grid_lines(ax, poly_rot, W, H)

    # Interior cells
    for c in int_cells_rot:
        coords = np.array(c.exterior.coords[:-1])
        draw_poly_patch(ax, coords,
                        fc=C_INTERIOR, ec="#2c7bb6", alpha=ALPHA_CELL,
                        lw=0.6, zorder=2)

    # Boundary cells
    for c in bnd_cells_rot:
        coords = np.array(c.exterior.coords[:-1])
        draw_poly_patch(ax, coords,
                        fc=C_BOUNDARY, ec="#d94801", alpha=ALPHA_CELL,
                        lw=0.6, zorder=2)

    # Rotated polygon outline on top
    draw_poly_patch(ax, poly_rot_coords,
                    fc="none", ec=C_POLY, alpha=1.0, lw=2.0, zorder=3,
                    label="$P_{-\\phi}$")
    c = poly_rot.centroid
    ax.text(c.x, c.y + 25, "$P_{-\\phi}$", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_POLY)

    # Axis labels showing the rotation has happened
    ax.axhline(0, color="#ccc", lw=0.6, ls="--", zorder=0)
    ax.axvline(0, color="#ccc", lw=0.6, ls="--", zorder=0)

    # Rotation annotation (curved arrow at bottom-left corner)
    all_rot_pts = poly_rot_coords
    bx = all_rot_pts[:, 0].min() - 60
    by = all_rot_pts[:, 1].min() - 40
    rotation_arc(ax, bx, by, r=50, phi_deg=phi,
                 color="#e41a1c", label=f"$-\\phi$")

    set_equal_ax(ax, poly_rot_coords, margin=90)
    ax.set_title("(b) Rotated frame: axis-aligned grid", fontsize=10, pad=6)

    # ════════════════════════════════════════════════════════════════
    # Panel (c) – back in original frame
    # ════════════════════════════════════════════════════════════════
    ax = axes[2]

    # Interior cells
    for coords in int_cells_orig:
        draw_poly_patch(ax, coords,
                        fc=C_INTERIOR, ec="#2c7bb6", alpha=ALPHA_CELL,
                        lw=0.6, zorder=2)
    # Boundary cells
    for coords in bnd_cells_orig:
        draw_poly_patch(ax, coords,
                        fc=C_BOUNDARY, ec="#d94801", alpha=ALPHA_CELL,
                        lw=0.6, zorder=2)

    # Polygon outline
    draw_poly_patch(ax, POLY_COORDS,
                    fc="none", ec=C_POLY, alpha=1.0, lw=2.0, zorder=3)
    c = poly_orig.centroid
    ax.text(c.x, c.y + 20, "$P$", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_POLY)

    # Rotation annotation
    bx = POLY_COORDS[:, 0].min() - 70
    by = POLY_COORDS[:, 1].min() - 40
    rotation_arc(ax, bx, by, r=50, phi_deg=phi,
                 color="#e41a1c", label=f"$+\\phi$")

    # Legend
    leg_handles = [
        mpatches.Patch(facecolor=C_INTERIOR, edgecolor="#2c7bb6",
                       alpha=0.7, label="Interior cell"),
        mpatches.Patch(facecolor=C_BOUNDARY, edgecolor="#d94801",
                       alpha=0.7, label="Boundary cell"),
        mpatches.Patch(facecolor="none",  edgecolor=C_POLY,
                       linewidth=1.8,     label="Polygon $P$"),
    ]
    ax.legend(handles=leg_handles, loc="lower right",
              fontsize=10, framealpha=0.85, edgecolor="#ccc")

    set_equal_ax(ax, POLY_COORDS, margin=80)
    ax.set_title("(c) Retained cells in original frame", fontsize=10, pad=6)

    # ── Save ──────────────────────────────────────────────────────────────────
    # bbox_inches='tight' trims the large default outer margins; pad_inches gives
    # a small uniform border so nothing touches the edge.
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    make_figure()
