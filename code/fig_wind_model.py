"""
fig_wind_model.py
=================
Generates paper/figures/fig_wind_model.pdf

Two-panel illustration of the wind-aware travel-time model.

  Panel (a) – Cell-to-cell transition geometry
      A small grid of cells with centres marked.  A red arrow shows
      a centre-to-centre transition (p_i -> p_j) labelled d_ij.  A green
      arrow shows the wind vector v_w.  A dashed reference arm and an arc
      at p_i annotate the angle theta_ij between heading and wind.  The
      formula t_ij = d_ij / (v_a + v_w cos theta_ij) is shown below.

  Panel (b) – Ground speed vs. heading angle
      Cartesian plot of  vg(theta) = v_a + v_w cos(theta)  for the
      numeric example  v_a = 12 m/s, v_w = 3 m/s.  Tailwind (theta=0),
      crosswind (theta=90/270), and headwind (theta=180) values are
      labelled.  A shaded band between the wind and no-wind curves
      emphasises the asymmetric speed benefit / penalty.

Usage
-----
    python code/fig_wind_model.py
"""

from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc
from matplotlib.lines import Line2D

OUT = Path(__file__).parent.parent / "paper" / "figures" / "fig_wind_model.pdf"

VA = 12.0   # UAV airspeed  (m/s)
VW = 3.0    # Wind speed    (m/s)

# Separate parameters for the velocity-triangle illustration in panel (a).
# A larger wind-speed ratio is used so the crab angle α and wind angle θ_ij
# are visually distinguishable; the polar plot in panel (b) still uses VA/VW.
VW_TRI = 6.0            # illustrative wind for the triangle (v_w/v_a = 0.5)
THETA_TRI_DEG = 50.0   # track→wind angle for the triangle

# Angle between the desired track direction and the wind (used for the illustration)
THETA_EXAMPLE_DEG = THETA_TRI_DEG

# Derived quantities for the velocity triangle inset
_th = math.radians(THETA_EXAMPLE_DEG)
_w_par  = VW_TRI * math.cos(_th)                          # along-track wind
_w_perp = VW_TRI * math.sin(_th)                          # cross-track wind
_vg_ex  = _w_par + math.sqrt(VA**2 - _w_perp**2)     # correct ground speed
_alpha  = math.degrees(math.asin(_w_perp / VA))       # crab angle (degrees)

# ── Colour palette (consistent with other figures) ───────────────────────────
C_CELL_BG  = "#f5f9fc"
C_GRID     = "#aaaaaa"
C_PATH     = "#d7191c"   # red  – path arrow, headwind labels
C_WIND     = "#1a9641"   # green – wind
C_DOT      = "#555555"
C_BLUE     = "#2c7bb6"   # blue  – ground-speed curve

WIND_DEG = 30.0          # wind direction: 30° CCW from East (NNE)
WIND_RAD = math.radians(WIND_DEG)

# ── Figure layout ─────────────────────────────────────────────────────────────
# figsize=7.0in → output PDF 504pt; LaTeX scales to textwidth 390pt (0.774×).
# fontsize 13pt × 0.774 ≈ 10pt in the printed PDF.
fig = plt.figure(figsize=(7.0, 3.4))
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.88, wspace=0.42)

# =============================================================================
# Panel (a):  Cell-to-cell transition + velocity-triangle inset
# =============================================================================
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_aspect("equal")
ax1.set_axis_off()

S = 1.0                 # normalised cell size
NR, NC = 3, 3           # grid rows × columns (y × x)


def cen(col, row):
    """Return centre of cell (col, row)."""
    return (col * S + S / 2, row * S + S / 2)


# ── Draw grid cells ───────────────────────────────────────────────────────────
for r in range(NR):
    for c in range(NC):
        rect = plt.Rectangle((c * S, r * S), S, S,
                              fc=C_CELL_BG, ec=C_GRID, lw=0.9, zorder=1)
        ax1.add_patch(rect)

# ── Cell-centre dots ──────────────────────────────────────────────────────────
for r in range(NR):
    for c in range(NC):
        ax1.plot(*cen(c, r), 'o', color=C_DOT, ms=3.5, zorder=4, clip_on=False)

# ── Source cell c_i = (0, 1) and destination c_j = (2, 1) ───────────────────
CI = (0, 1)   # column, row
CJ = (2, 1)
pxi, pyi = cen(*CI)
pxj, pyj = cen(*CJ)

# Highlight source and destination cells
for col, row in [CI, CJ]:
    ax1.add_patch(plt.Rectangle((col * S, row * S), S, S,
                                 fc="#fde8e8", ec=C_PATH, lw=1.4, zorder=2))

ax1.plot(pxi, pyi, 'o', color=C_PATH, ms=6.5, zorder=8)
ax1.plot(pxj, pyj, 'o', color=C_PATH, ms=6.5, zorder=8)

ax1.text(pxi - 0.14, pyi - 0.18, r"$p_i$",
         fontsize=10, color=C_PATH, ha='center', va='top')
ax1.text(pxj + 0.14, pyj - 0.18, r"$p_j$",
         fontsize=10, color=C_PATH, ha='center', va='top')

# ── Desired ground track p_i → p_j (red dashed arrow) ────────────────────────
ax1.annotate("", xy=(pxj, pyj), xytext=(pxi, pyi),
             arrowprops=dict(arrowstyle="-|>", color=C_PATH, lw=1.6,
                             linestyle="dashed"), zorder=7)

# d_ij label
ax1.text((pxi + pxj) / 2, (pyi + pyj) / 2 + 0.42,
         r"$d_{ij}$  (ground track $\hat{u}_{ij}$)",
         fontsize=10, color=C_PATH, ha='center', va='bottom')

# ──────────────────────────────────────────────────────────────────────────────
# Velocity-triangle inset above the grid
# The illustration uses THETA_EXAMPLE_DEG = 30° between track (+x) and wind
# Track direction: +x  (same direction as p_i → p_j)
# Crab heading: -alpha below +x (drone noses into the crosswind)
# ──────────────────────────────────────────────────────────────────────────────
SCALE = 1.6 / VA    # scale so v_a maps to 1.6 grid units (larger triangle)

# Vector components in grid-unit space
alpha_rad = math.radians(_alpha)
va_x =  SCALE * VA * math.cos(-alpha_rad)   # slightly below +x
va_y =  SCALE * VA * math.sin(-alpha_rad)
vw_x =  SCALE * VW_TRI * math.cos(_th)          # wind at THETA_TRI_DEG from +x
vw_y =  SCALE * VW_TRI * math.sin(_th)
vg_x =  SCALE * _vg_ex                      # along +x
vg_y =  0.0

# Triangle origin: slightly left of the top-centre of the panel
tx = NC * S / 2 - vg_x / 2 - 0.15
ty = NR * S + 1.25

# v_g  (red dashed, along track)
ax1.annotate("", xy=(tx + vg_x, ty + vg_y), xytext=(tx, ty),
             arrowprops=dict(arrowstyle="-|>", color=C_PATH, lw=1.8,
                             linestyle="dashed"), zorder=10)
ax1.text(tx + vg_x / 2 + 1.9, ty + 0.53,
         r"$\mathbf{v}_g$ (ground velocity)",
         fontsize=10, color=C_PATH, ha='center', va='top')

# v_a  (blue solid, crab heading)
ax1.annotate("", xy=(tx + va_x, ty + va_y), xytext=(tx, ty),
             arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=2.2), zorder=10)
ax1.text(tx + va_x / 2 + 1.20, ty + va_y / 2 - 0.75,
         r"$\mathbf{v}_a$ (airspeed)",
         fontsize=10, color=C_BLUE, ha='right', va='bottom')

# v_w  (green, from tip of v_a closing to tip of v_g)
ax1.annotate("", xy=(tx + vg_x, ty), xytext=(tx + va_x, ty + va_y),
             arrowprops=dict(arrowstyle="-|>", color=C_WIND, lw=2.0), zorder=10)
ax1.text(tx + va_x + (vg_x - va_x) / 2 + 0.25,
         ty + va_y / 2 - 0.20,
         r"$\mathbf{v}_w$ (wind)",
         fontsize=10, color=C_WIND, ha='left', va='center')

# Crab angle arc (α, between v_g and v_a at the triangle origin)
arc_r_alpha = 0.58
theta_alpha = np.linspace(-alpha_rad, 0, 40)
ax1.plot(tx + arc_r_alpha * np.cos(theta_alpha),
         ty + arc_r_alpha * np.sin(theta_alpha),
         color=C_BLUE, lw=1.4, zorder=11)
bis_alpha = math.radians(-_alpha / 2)
ax1.text(tx + (arc_r_alpha + 0.18) * math.cos(bis_alpha),
         ty + (arc_r_alpha + 0.18) * math.sin(bis_alpha),
         r"$\alpha$", fontsize=10, color=C_BLUE, ha='center', va='center')

# θ_ij arc (between track v_g and wind direction, at the triangle origin)
arc_r_th = 0.58
theta_th = np.linspace(0, _th, 50)
ax1.plot(tx + arc_r_th * np.cos(theta_th),
         ty + arc_r_th * np.sin(theta_th),
         color=C_WIND, lw=1.4, ls='--', zorder=11)
bis_th = _th / 2
ax1.text(tx + (arc_r_th + 0.18) * math.cos(bis_th),
         ty + (arc_r_th + 0.18) * math.sin(bis_th),
         r"$\theta_{ij}$", fontsize=10, color=C_WIND, ha='center', va='center')

# equation: v_a + v_w = v_g
#ax1.text(tx + vg_x + 0.18, ty,
#         r"$\mathbf{v}_a+\mathbf{v}_w=\mathbf{v}_g$",
#         fontsize=10, color="#333333", va='center', ha='left')

# ── Formula text box (below grid) ────────────────────────────────────────────
ax1.text(NC * S / 2, -0.9,
         r"$t_{ij} = \dfrac{d_{ij}}{v_w\cos\theta_{ij}+\sqrt{v_a^2-v_w^2\sin^2\theta_{ij}}}$",
         ha='center', va='center', fontsize=10,
         bbox=dict(fc="white", ec="#aaaaaa", boxstyle="round,pad=0.45"), zorder=12)

ax1.set_xlim(-0.22, NC * S + 1.00)
ax1.set_ylim(-0.95, NR * S + 2.50)
ax1.set_title("(a)  Velocity triangle and transition geometry\n"
              r"(triangle schematic: $v_w/v_a=0.5$)", fontsize=10, pad=6)

# =============================================================================
# Panel (b):  Ground speed vs. heading angle (Cartesian)
# =============================================================================
ax2 = fig.add_subplot(1, 2, 2)

theta_deg = np.linspace(0, 360, 720)
theta_rad = np.radians(theta_deg)
vg        = VW * np.cos(theta_rad) + np.sqrt(np.maximum(0, VA**2 - VW**2 * np.sin(theta_rad)**2))
vg_ref    = np.full_like(vg, VA)

# ── Shading ───────────────────────────────────────────────────────────────────
ax2.fill_between(theta_deg, vg_ref, vg,
                 where=(vg >= vg_ref), color=C_WIND, alpha=0.15, label="_nolegend_")
ax2.fill_between(theta_deg, vg_ref, vg,
                 where=(vg < vg_ref), color=C_PATH, alpha=0.15, label="_nolegend_")

# ── Curves ────────────────────────────────────────────────────────────────────
ax2.plot(theta_deg, vg,
         color=C_BLUE, lw=2.0,
         label=rf"$v_g(\theta)=v_w\cos\theta+\sqrt{{v_a^2-v_w^2\sin^2\theta}}$" + "\n" +
               rf"($v_a={VA:.0f}$ m/s, $v_w={VW:.0f}$ m/s)")
ax2.axhline(VA, color="#888888", lw=1.3, ls="--",
            label=rf"No-wind reference $v_a={VA:.0f}$ m/s")

# ── Key annotations ───────────────────────────────────────────────────────────
# Tailwind (theta = 0°)
ax2.plot(0, VA + VW, 'o', color=C_WIND, ms=7, zorder=5)
ax2.annotate(f"Tailwind $v_g = {VA+VW:.0f}$ m/s",
             xy=(0, VA + VW), xytext=(20, VA + VW - 1.5),
             fontsize=10, color=C_WIND, va='bottom',
             arrowprops=dict(arrowstyle='->', color=C_WIND, lw=0.9))

# Headwind (theta = 180°)
ax2.plot(180, VA - VW, 'o', color=C_PATH, ms=7, zorder=5)
ax2.annotate(f"Headwind\n$v_g = {VA-VW:.0f}$ m/s",
             xy=(180, VA - VW), xytext=(205, VA - VW - 0.5),
             fontsize=10, color=C_PATH, va='top',
             arrowprops=dict(arrowstyle='->', color=C_PATH, lw=0.9))

# Crosswind (theta = 90°)
vg_crosswind = math.sqrt(VA**2 - VW**2)   # = sqrt(135) ≈ 11.619
ax2.plot(90, vg_crosswind, 's', color="#888888", ms=5.5, zorder=5)
ax2.annotate(rf"Crosswind  $v_g=\sqrt{{v_a^2-v_w^2}}\approx{vg_crosswind:.1f}$ m/s",
             xy=(90, vg_crosswind), xytext=(25, vg_crosswind + 0.7),
             fontsize=10, color="#555555", va='bottom',
             arrowprops=dict(arrowstyle='->', color="#888888", lw=0.9))

# ── Axes ──────────────────────────────────────────────────────────────────────
ax2.set_xlabel(r"Heading angle $\theta$ relative to wind ($^\circ$)", fontsize=10)
ax2.set_ylabel("Ground speed $v_g$ (m/s)", fontsize=10)
ax2.set_xlim(0, 360)
ax2.set_ylim(VA - VW - 1.8, VA + VW + 2.0)
ax2.set_xticks([0, 90, 180, 270, 360])
ax2.set_xticklabels([r"$0^\circ$", r"$90^\circ$", r"$180^\circ$",
                     r"$270^\circ$", r"$360^\circ$"])
ax2.set_yticks([VA - VW, VA, VA + VW])
ax2.set_yticklabels([f"{VA-VW:.0f}", f"{VA:.0f}", f"{VA+VW:.0f}"])
ax2.tick_params(labelsize=10)
ax2.grid(alpha=0.30, lw=0.6)
ax2.legend(loc="upper right", fontsize=8, frameon=True)
ax2.set_title(r"(b)  Ground speed vs. heading angle", fontsize=10, pad=6)

# =============================================================================
# Save
# =============================================================================
# Save at exact figsize (no bbox_inches='tight') so fonts scale predictably.
fig.savefig(str(OUT))
print(f"Saved {OUT}")
