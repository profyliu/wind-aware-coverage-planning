"""
fig_energy_unimodal.py
Generates paper/figures/fig_energy_unimodal.pdf

Two-panel figure illustrating unimodality of the strip-leg energy function
E_ell(v_g) = P(v_a(v_g, theta)) / v_g  [J/m]

Panel (a): fixed wind ratio rho = 0.25, five headings
Panel (b): fixed crosswind heading theta = 90 deg, five wind ratios

Saves fig to paper/figures/fig_energy_unimodal.pdf (and .png for preview).
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------
# Power model parameters (DJI Phantom 4 class, Zeng et al. 2019)
# ------------------------------------------------------------------
P0   = 79.86    # blade-profile hover power  [W]
Pi   = 88.63    # induced hover power        [W]
Utip = 120.0    # blade-tip speed            [m/s]
v0   = 4.03     # hover induced velocity     [m/s]
rho  = 1.225    # air density                [kg/m^3]
Cd   = 0.6      # fuselage drag coefficient
A    = 0.0151   # reference frontal area     [m^2]


def power(va: float) -> float:
    """Zeng et al. rotary-wing power model."""
    blade_profile = P0 * (1 + 3 * va**2 / Utip**2)
    induced = Pi * math.sqrt(math.sqrt(1 + va**4 / (4 * v0**4)) - va**2 / (2 * v0**2))
    parasitic = 0.5 * rho * Cd * A * va**3
    return blade_profile + induced + parasitic


def groundspeed(va: float, theta_rad: float, vw: float) -> float:
    """Ground speed for airspeed va at heading theta (rad from wind)."""
    w_along = vw * math.cos(theta_rad)
    w_perp_sq = vw**2 * math.sin(theta_rad)**2
    disc = va**2 - w_perp_sq
    if disc < 0:
        return math.nan
    return w_along + math.sqrt(disc)


def energy_per_m(va: float, theta_rad: float, vw: float) -> float:
    """Energy per unit distance [J/m] at airspeed va."""
    vg = groundspeed(va, theta_rad, vw)
    if math.isnan(vg) or vg <= 0:
        return math.nan
    return power(va) / vg


def find_optimal_va(theta_rad: float, vw: float,
                    va_min: float = 6.0, va_max: float = 28.0,
                    tol: float = 1e-4) -> tuple[float, float]:
    """Golden-section search for minimum energy-per-m over airspeed."""
    gr = (math.sqrt(5) - 1) / 2
    a, b = max(va_min, math.sqrt((vw * math.sin(theta_rad))**2) + 0.01), va_max
    if a >= b:
        return a, energy_per_m(a, theta_rad, vw)
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    while abs(b - a) > tol:
        if energy_per_m(c, theta_rad, vw) < energy_per_m(d, theta_rad, vw):
            b = d
        else:
            a = c
        c = b - gr * (b - a)
        d = a + gr * (b - a)
    va_opt = (a + b) / 2
    return va_opt, energy_per_m(va_opt, theta_rad, vw)


# ------------------------------------------------------------------
# Plot setup
# ------------------------------------------------------------------
VA = np.linspace(6.0, 28.0, 500)
VA_BOUNDS = (8.0, 25.0)       # experimental bounds for shading

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

HEADINGS_DEG = [0, 45, 90, 135, 180]
HEADING_LABELS = [r"$\theta=0°$ (headwind)", r"$\theta=45°$",
                  r"$\theta=90°$ (crosswind)", r"$\theta=135°$",
                  r"$\theta=180°$ (tailwind)"]
HEADING_COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]

WIND_RATIOS = [0.0, 0.10, 0.20, 0.30, 0.40]
WIND_LABELS = [r"$\rho=0.0$ (calm)", r"$\rho=0.1$", r"$\rho=0.2$",
               r"$\rho=0.3$", r"$\rho=0.4$"]
WIND_COLORS = ["#aec7e8", "#6baed6", "#3182bd", "#08519c", "#08306b"]

AIRSPEED_NOMINAL = 12.0   # m/s (baseline airspeed)

# ---- Panel (a): vary heading, fixed rho = 0.25, va_nominal = 12 m/s ----
ax = axes[0]
vw_fixed = 0.25 * AIRSPEED_NOMINAL   # rho = 0.25

for deg, label, color in zip(HEADINGS_DEG, HEADING_LABELS, HEADING_COLORS):
    theta = math.radians(deg)
    vals = [energy_per_m(va, theta, vw_fixed) for va in VA]
    ax.plot(VA, vals, color=color, lw=1.6, label=label)
    va_opt, e_opt = find_optimal_va(theta, vw_fixed, *VA_BOUNDS)
    ax.plot(va_opt, e_opt, "o", color=color, ms=5, zorder=5)

ax.axvspan(*VA_BOUNDS, alpha=0.44, color="gray", label=r"$[v_a^\mathrm{min},\,v_a^\mathrm{max}]$")
ax.axvline(AIRSPEED_NOMINAL, color="black", lw=0.8, ls="--", label=r"$v_a=12\,\mathrm{m/s}$ baseline")
ax.set_xlabel("Airspeed $v_a$ (m/s)", fontsize=11)
ax.set_ylabel("Energy per unit distance (J/m)", fontsize=11)
ax.set_title(r"(a) Vary heading, $\rho = 0.25$", fontsize=11)
ax.legend(fontsize=9.5, loc="upper right")
ax.set_xlim(VA[0], VA[-1])
ax.tick_params(labelsize=9)

# ---- Panel (b): vary wind ratio, fixed crosswind theta = 90 deg ----
ax = axes[1]
theta_fixed = math.radians(90)

for ratio, label, color in zip(WIND_RATIOS, WIND_LABELS, WIND_COLORS):
    vw = ratio * AIRSPEED_NOMINAL
    vals = [energy_per_m(va, theta_fixed, vw) for va in VA]
    ax.plot(VA, vals, color=color, lw=1.6, label=label)
    va_opt, e_opt = find_optimal_va(theta_fixed, vw, *VA_BOUNDS)
    ax.plot(va_opt, e_opt, "o", color=color, ms=5, zorder=5)

ax.axvspan(*VA_BOUNDS, alpha=0.44, color="gray", label=r"$[v_a^\mathrm{min},\,v_a^\mathrm{max}]$")
ax.axvline(AIRSPEED_NOMINAL, color="black", lw=0.8, ls="--", label=r"$v_a=12\,\mathrm{m/s}$ baseline")
ax.set_xlabel("Airspeed $v_a$ (m/s)", fontsize=11)
ax.set_title(r"(b) Vary wind ratio, $\theta = 90°$ (crosswind)", fontsize=11)
ax.legend(fontsize=9.5, loc="upper right")
ax.set_xlim(VA[0], VA[-1])
ax.tick_params(labelsize=9)
ax.set_yticklabels([])

fig.tight_layout()

out_dir = Path(__file__).resolve().parent.parent / "paper" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "fig_energy_unimodal.pdf", bbox_inches="tight")
fig.savefig(out_dir / "fig_energy_unimodal.png", dpi=150, bbox_inches="tight")
print(f"Saved to {out_dir / 'fig_energy_unimodal.pdf'}")
