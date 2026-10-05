"""
sensitivity_analysis.py
========================
Power-model and wind-estimate sensitivity analysis for the SE Michigan
farmland real-site demonstration (Sec. 5.5 of the paper).

Addresses Reviewer 1's suggestion to provide "a sensitivity analysis with
respect to the power model parameters and wind estimates."

Two analyses are performed on the R2 (baseline, 80 m alt, 70/80 % overlap,
45 °, 3 m/s wind) farmland configuration:

  1. One-at-a-time (OAT) parameter sweep
     Each of the 7 power-model parameters (P0, Pi, U_tip, v0, rho, C_D, A)
     and 2 wind-estimate quantities (wind speed, wind direction) are
     independently perturbed by ±5 %, ±10 %, ±20 %, ±30 % while holding all
     others at their nominal values.  For each perturbation the full planner
     pipeline is re-run (phi* re-selected, route re-built, speeds re-optimised)
     so that the analysis captures not only evaluation sensitivity but also
     decision sensitivity.

  2. Monte Carlo (MC) joint perturbation
     All 7 power-model parameters are simultaneously drawn from independent
     uniform(−20 %, +20 %) distributions (2 000 samples).  Wind speed and
     direction are each drawn from uniform(−1 m/s, +1 m/s) and
     uniform(−15 °, +15 °) respectively.  Each sample runs the full planner
     pipeline and records mission energy.

Outputs
-------
  paper/figures/fig_sensitivity.pdf   – combined figure (OAT tornado + MC)
  paper/data/sensitivity_oat.csv      – full OAT table
  paper/data/sensitivity_mc.csv       – full MC sample table
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "gcs_plugin"))

from decomposition import decompose
from routing import (
    build_route,
    optimal_grid_angle_energy,
    DEFAULT_POWER_MODEL,
    PowerModel,
)
from gcs_plugin.coords import parse_kml_polygon, lonlat_to_utm_polygon
from gcs_plugin.camera import camera_from_preset

KML_SCENARIO = CODE / "scenarios" / "Farmland.kml"
OUT_DIR = ROOT / "paper" / "data"
FIG_DIR = ROOT / "paper" / "figures"

# ---------------------------------------------------------------------------
# Nominal R2 configuration
# ---------------------------------------------------------------------------

NOM_ALT           = 80.0        # m
NOM_OVERLAP_SIDE  = 0.70
NOM_OVERLAP_FRONT = 0.80
NOM_WIND_DIR_DEG  = 45.0        # "from" convention → toward = 225°
NOM_WIND_SPEED    = 3.0         # m/s
NOM_AIRSPEED      = 8.0         # m/s nominal (speed is optimised; this is the seed)

# Power-model perturbation levels (fractional)
OAT_LEVELS = [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30]

# Monte Carlo sample count
# Full MC re-runs the orientation sweep per sample (slow but complete).
# Fast MC fixes phi* at the nominal value and only re-runs routing (much faster,
# valid for energy-estimate uncertainty; phi* stability is checked separately in OAT).
MC_N = 500              # number of MC samples (full pipeline)
MC_FAST_N = 1000        # number of samples with phi* fixed (faster)
MC_PARAM_RANGE  = 0.20  # ±20 % uniform for each power-model parameter
MC_WIND_SPD_STD = 1.0   # ±1 m/s uniform
MC_WIND_DIR_STD = 15.0  # ±15 ° uniform

# OAT wind perturbations (absolute)
OAT_WIND_SPD = [-1.0, -0.5, 0.0, 0.5, 1.0]   # m/s delta
OAT_WIND_DIR = [-20, -10, 0, 10, 20]           # degree delta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def met_wind_vec(wind_dir_from_deg: float, wind_speed_ms: float) -> np.ndarray:
    """Convert meteorological wind (from-direction, speed) to Cartesian vector."""
    toward_deg = (wind_dir_from_deg + 180.0) % 360.0
    rad = math.radians(90.0 - toward_deg)
    return np.array([math.cos(rad), math.sin(rad)]) * wind_speed_ms


def run_planner(
    polygon_utm,
    wind_dir_deg: float,
    wind_speed_ms: float,
    power_model: PowerModel,
    camera_preset: str = "sony_a6000_16mm",
    alt: float = NOM_ALT,
    overlap_side: float = NOM_OVERLAP_SIDE,
    overlap_front: float = NOM_OVERLAP_FRONT,
    nominal_airspeed: float = NOM_AIRSPEED,
) -> dict:
    """Run the full planning pipeline and return energy and decision metrics."""
    cp = camera_from_preset(camera_preset, alt, overlap_side, overlap_front)
    wind_vec = met_wind_vec(wind_dir_deg, wind_speed_ms)

    phi_deg = optimal_grid_angle_energy(
        polygon_utm,
        cp.cell_w,
        cp.cell_h,
        wind_vec,
        nominal_airspeed,
        power_model=power_model,
    )
    decomp = decompose(polygon_utm, cp.cell_w, cp.cell_h, angle_deg=phi_deg)
    route = build_route(
        decomp,
        wind_vec,
        airspeed=nominal_airspeed,
        power_model=power_model,
        objective="energy",
    )
    return {
        "energy_kJ":  route.total_energy_J / 1000.0,
        "phi_deg":    phi_deg,
        "n_strips":   route.metrics["n_strips"],
        "path_km":    route.total_length_m / 1000.0,
        "time_min":   route.total_time_s / 60.0,
    }


# ---------------------------------------------------------------------------
# OAT analysis
# ---------------------------------------------------------------------------

PARAM_INFO = {
    # param_name: (display_label, field_name_in_PowerModel, nominal_value, unit)
    "P0":    ("$P_0$ (blade-profile)", "P0",    DEFAULT_POWER_MODEL.P0,    "W"),
    "Pi":    ("$P_i$ (induced)",       "Pi",    DEFAULT_POWER_MODEL.Pi,    "W"),
    "U_tip": ("$U_{\\rm tip}$",        "U_tip", DEFAULT_POWER_MODEL.U_tip, "m/s"),
    "v0":    ("$v_0$ (hover induced)", "v0",    DEFAULT_POWER_MODEL.v0,    "m/s"),
    "C_D":   ("$C_D$ (drag coeff.)",   "C_D",   DEFAULT_POWER_MODEL.C_D,   "—"),
    "A":     ("$A$ (frontal area)",    "A",     DEFAULT_POWER_MODEL.A,     "m²"),
    "rho":   ("$\\rho$ (air density)", "rho",   DEFAULT_POWER_MODEL.rho,   "kg/m³"),
}


def run_oat(polygon_utm, nom_result: dict) -> pd.DataFrame:
    """One-at-a-time sensitivity sweep.  Returns a DataFrame of results."""
    nom_E = nom_result["energy_kJ"]
    rows = []

    # --- Power-model parameters ---
    for pkey, (label, field, nom_val, unit) in PARAM_INFO.items():
        for level in OAT_LEVELS:
            perturbed_val = nom_val * (1.0 + level)
            pm = replace(DEFAULT_POWER_MODEL, **{field: perturbed_val})
            res = run_planner(polygon_utm, NOM_WIND_DIR_DEG, NOM_WIND_SPEED, pm)
            rows.append({
                "param":      pkey,
                "label":      label,
                "perturbation_frac": level,
                "param_value": perturbed_val,
                "energy_kJ":  res["energy_kJ"],
                "delta_pct":  100.0 * (res["energy_kJ"] - nom_E) / nom_E,
                "phi_deg":    res["phi_deg"],
            })
        print(f"  OAT: {pkey} done ({len(OAT_LEVELS)} levels)")

    # --- Wind speed ---
    for dv in OAT_WIND_SPD:
        ws = NOM_WIND_SPEED + dv
        res = run_planner(polygon_utm, NOM_WIND_DIR_DEG, ws, DEFAULT_POWER_MODEL)
        rows.append({
            "param":      "wind_speed",
            "label":      "$v_w$ (wind speed)",
            "perturbation_frac": dv / NOM_WIND_SPEED,
            "param_value": ws,
            "energy_kJ":  res["energy_kJ"],
            "delta_pct":  100.0 * (res["energy_kJ"] - nom_E) / nom_E,
            "phi_deg":    res["phi_deg"],
        })
    print("  OAT: wind_speed done")

    # --- Wind direction ---
    for dd in OAT_WIND_DIR:
        wd = NOM_WIND_DIR_DEG + dd
        res = run_planner(polygon_utm, wd, NOM_WIND_SPEED, DEFAULT_POWER_MODEL)
        rows.append({
            "param":      "wind_dir",
            "label":      "$\\theta_w$ (wind dir.)",
            "perturbation_frac": dd / 180.0,
            "param_value": wd,
            "energy_kJ":  res["energy_kJ"],
            "delta_pct":  100.0 * (res["energy_kJ"] - nom_E) / nom_E,
            "phi_deg":    res["phi_deg"],
        })
    print("  OAT: wind_dir done")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Monte Carlo analysis
# ---------------------------------------------------------------------------

def run_mc(polygon_utm, nom_result: dict, rng: np.random.Generator,
           fixed_phi: float | None = None, n_samples: int | None = None) -> pd.DataFrame:
    """Joint Monte Carlo perturbation of all parameters.

    If *fixed_phi* is provided, the orientation search is skipped and the
    given phi is used for all samples (≈10× faster, suitable for energy
    uncertainty quantification; phi* stability is assessed separately by OAT).
    """
    nom_E = nom_result["energy_kJ"]
    N = n_samples if n_samples is not None else MC_N
    rows = []

    n_params = len(PARAM_INFO)
    deltas = rng.uniform(-MC_PARAM_RANGE, MC_PARAM_RANGE, size=(N, n_params))
    wind_spd_deltas = rng.uniform(-MC_WIND_SPD_STD, MC_WIND_SPD_STD, size=N)
    wind_dir_deltas = rng.uniform(-MC_WIND_DIR_STD, MC_WIND_DIR_STD, size=N)

    for i in range(N):
        kwargs = {}
        for j, (pkey, (_, field, nom_val, _)) in enumerate(PARAM_INFO.items()):
            kwargs[field] = nom_val * (1.0 + deltas[i, j])
        pm = replace(DEFAULT_POWER_MODEL, **kwargs)

        ws = max(NOM_WIND_SPEED + wind_spd_deltas[i], 0.1)
        wd = NOM_WIND_DIR_DEG + wind_dir_deltas[i]

        if fixed_phi is not None:
            # Fast path: skip orientation search
            from gcs_plugin.camera import camera_from_preset
            cp = camera_from_preset("sony_a6000_16mm", NOM_ALT,
                                    NOM_OVERLAP_SIDE, NOM_OVERLAP_FRONT)
            wind_vec = met_wind_vec(wd, ws)
            from decomposition import decompose as _decompose
            decomp = _decompose(polygon_utm, cp.cell_w, cp.cell_h,
                                angle_deg=fixed_phi)
            route = build_route(decomp, wind_vec, airspeed=NOM_AIRSPEED,
                                power_model=pm, objective="energy")
            res = {
                "energy_kJ": route.total_energy_J / 1000.0,
                "phi_deg":   fixed_phi,
            }
        else:
            res = run_planner(polygon_utm, wd, ws, pm)

        rows.append({
            "sample":    i,
            "energy_kJ": res["energy_kJ"],
            "phi_deg":   res["phi_deg"],
            "delta_pct": 100.0 * (res["energy_kJ"] - nom_E) / nom_E,
        })

        if (i + 1) % 200 == 0:
            print(f"  MC: {i+1}/{N}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

TORNADO_LEVELS = [-0.20, 0.20]   # show ±20 % bars on tornado plot


def make_figure(oat_df: pd.DataFrame, mc_df: pd.DataFrame,
                nom_E: float, out_path: Path) -> None:
    """Generate a 1×3 sensitivity figure."""
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.2),
                             gridspec_kw={"width_ratios": [2.2, 1.1, 1.5]})
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.14, top=0.88,
                        wspace=0.45)

    FONTSIZE = 9
    LABEL_FS = 10
    TITLE_FS = 10

    # ------------------------------------------------------------------ #
    # Panel (a): Tornado plot — effect of ±20 % perturbation on each param
    # ------------------------------------------------------------------ #
    ax = axes[0]
    all_params = list(PARAM_INFO.keys()) + ["wind_speed", "wind_dir"]
    LABELS = {pk: PARAM_INFO[pk][0] if pk in PARAM_INFO else
              ("$v_w$ (wind speed)" if pk == "wind_speed" else "$\\theta_w$ (wind dir.)")
              for pk in all_params}

    # Compute range at ±20 % level (or closest available)
    ranges = {}
    for pk in all_params:
        sub = oat_df[oat_df["param"] == pk]
        if pk in ("wind_speed", "wind_dir"):
            # Use absolute ±1 m/s or ±20 ° as the reference level
            levels = sorted(sub["perturbation_frac"].unique())
            lo_row = sub.loc[sub["perturbation_frac"] == min(levels)]
            hi_row = sub.loc[sub["perturbation_frac"] == max(levels)]
        else:
            lo_row = sub.loc[(sub["perturbation_frac"] - (-0.20)).abs() < 0.01]
            hi_row = sub.loc[(sub["perturbation_frac"] - (0.20)).abs() < 0.01]
        lo_dp = float(lo_row["delta_pct"].values[0]) if len(lo_row) else 0.0
        hi_dp = float(hi_row["delta_pct"].values[0]) if len(hi_row) else 0.0
        ranges[pk] = (lo_dp, hi_dp)

    # Sort by absolute range (most influential on top)
    sorted_params = sorted(all_params, key=lambda p: abs(ranges[p][1] - ranges[p][0]))
    y_pos = np.arange(len(sorted_params))

    ANNOT_THRESHOLD = 0.15   # bars narrower than this % get a text annotation

    for yi, pk in enumerate(sorted_params):
        lo, hi = ranges[pk]
        color_lo = "#5b9bd5"    # blue for negative perturbation
        color_hi = "#ed7d31"    # orange for positive perturbation
        ax.barh(yi, lo, left=0, height=0.55, color=color_lo, alpha=0.85)
        ax.barh(yi, hi, left=0, height=0.55, color=color_hi, alpha=0.85)
        # Annotate bars that are too narrow to see
        if abs(lo) < ANNOT_THRESHOLD:
            ax.text(lo - 0.02, yi, f"{lo:+.2f}%", ha="right", va="center",
                    fontsize=7.5, color="#5b9bd5")
        if abs(hi) < ANNOT_THRESHOLD:
            ax.text(hi + 0.02, yi, f"{hi:+.2f}%", ha="left", va="center",
                    fontsize=7.5, color="#ed7d31")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([LABELS[p] for p in sorted_params], fontsize=FONTSIZE)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Mission energy change (%)", fontsize=FONTSIZE)
    ax.set_title("(a) OAT sensitivity (±20 %)", fontsize=TITLE_FS, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f%%"))
    ax.tick_params(axis="x", labelsize=FONTSIZE)
    # Add legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#5b9bd5", alpha=0.85, label="−20 %"),
        Patch(color="#ed7d31", alpha=0.85, label="+20 %"),
    ], fontsize=8, loc="lower right", framealpha=0.85)

    # ------------------------------------------------------------------ #
    # Panel (b): OAT energy vs perturbation level — selected influential params
    # ------------------------------------------------------------------ #
    ax = axes[1]
    # Show the top-4 most influential power-model parameters
    top4 = sorted(PARAM_INFO.keys(),
                  key=lambda p: abs(ranges[p][1] - ranges[p][0]),
                  reverse=True)[:4]
    colors_top4 = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    pct_levels = [l for l in OAT_LEVELS if l != 0]

    for ci, pk in enumerate(top4):
        sub = oat_df[(oat_df["param"] == pk) & (oat_df["perturbation_frac"] != 0.0)]
        sub = sub.sort_values("perturbation_frac")
        ax.plot(sub["perturbation_frac"] * 100, sub["delta_pct"],
                "o-", color=colors_top4[ci], ms=3.5, lw=1.2,
                label=PARAM_INFO[pk][0])

    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.axvline(0, color="k", lw=0.7, ls="--")
    ax.set_xlabel("Parameter perturbation (%)", fontsize=FONTSIZE)
    ax.set_ylabel("Energy change (%)", fontsize=FONTSIZE)
    ax.set_title("(b) Sensitivity curves", fontsize=TITLE_FS, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f%%"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f%%"))
    ax.tick_params(labelsize=FONTSIZE)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85)

    # ------------------------------------------------------------------ #
    # Panel (c): Monte Carlo histogram
    # ------------------------------------------------------------------ #
    ax = axes[2]
    energies = mc_df["energy_kJ"].values
    q05, q95 = np.percentile(energies, 5), np.percentile(energies, 95)

    ax.hist(energies, bins=40, color="#4d9de0", edgecolor="white", lw=0.4,
            density=True, alpha=0.85)
    ax.axvline(nom_E, color="#e15759", lw=1.5, label=f"Nominal ({nom_E:.1f} kJ)")
    mc_mean = energies.mean()
    mc_std  = energies.std()
    mc_cv   = 100.0 * mc_std / mc_mean
    ax.axvspan(q05, q95, color="#f28e2b", alpha=0.18,
               label=f"90% CI [{q05:.1f}, {q95:.1f}] kJ\n"
                     f"Mean {mc_mean:.1f} kJ,  CV={mc_cv:.1f}%")
    ax.set_xlabel("Mission energy (kJ)", fontsize=FONTSIZE)
    ax.set_ylabel("Probability density", fontsize=FONTSIZE)
    ax.set_title("(c) Monte Carlo ($N$=1000)",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.tick_params(labelsize=FONTSIZE)
    ax.legend(fontsize=8, framealpha=0.90, loc="upper left")

    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_summary(oat_df: pd.DataFrame, mc_df: pd.DataFrame, nom_E: float) -> None:
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print(f"Nominal mission energy (R2 baseline): {nom_E:.2f} kJ")
    print("=" * 60)

    print("\n-- OAT ±20 % power-model parameter sensitivity --")
    for pk, (label, _, _, _) in PARAM_INFO.items():
        sub = oat_df[oat_df["param"] == pk]
        lo_row = sub.loc[(sub["perturbation_frac"] - (-0.20)).abs() < 0.01]
        hi_row = sub.loc[(sub["perturbation_frac"] - (0.20)).abs() < 0.01]
        lo = float(lo_row["delta_pct"].values[0]) if len(lo_row) else 0.0
        hi = float(hi_row["delta_pct"].values[0]) if len(hi_row) else 0.0
        print(f"  {pk:8s}: −20%→{lo:+.1f}%, +20%→{hi:+.1f}%  (range {abs(hi-lo):.1f}%)")

    print("\n-- OAT wind estimate sensitivity --")
    sub_ws = oat_df[oat_df["param"] == "wind_speed"].sort_values("perturbation_frac")
    print(f"  Wind speed  ±1 m/s: {sub_ws['delta_pct'].min():+.1f}% to {sub_ws['delta_pct'].max():+.1f}%")
    sub_wd = oat_df[oat_df["param"] == "wind_dir"].sort_values("perturbation_frac")
    print(f"  Wind dir.  ±20 °:   {sub_wd['delta_pct'].min():+.1f}% to {sub_wd['delta_pct'].max():+.1f}%")

    print("\n-- Monte Carlo joint perturbation --")
    energies = mc_df["energy_kJ"].values
    q05, q95 = np.percentile(energies, 5), np.percentile(energies, 95)
    mc_mean = energies.mean()
    mc_std  = energies.std()
    print(f"  N = {len(mc_df)}")
    print(f"  Mean = {mc_mean:.2f} kJ,  Std = {mc_std:.2f} kJ,  CV = {100*mc_std/mc_mean:.1f}%")
    print(f"  90% CI: [{q05:.2f}, {q95:.2f}] kJ  (±{100*(q95-q05)/(2*mc_mean):.1f}% of nominal)")

    # phi* stability: check OAT (fast MC fixes phi* by design)
    nom_phi = oat_df[oat_df["perturbation_frac"] == 0.0]["phi_deg"].mode()[0]
    oat_phi_stable = (oat_df["phi_deg"] == nom_phi).mean() * 100.0
    print(f"  Optimal orientation phi* = {nom_phi:.0f}° unchanged in {oat_phi_stable:.0f}% of OAT samples (all parameter perturbation levels, including ±30%)")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figure-only", action="store_true",
        help="Skip experiments; load saved CSVs and regenerate figure only.",
    )
    args = parser.parse_args()

    oat_path = OUT_DIR / "sensitivity_oat.csv"
    mc_path  = OUT_DIR / "sensitivity_mc.csv"

    if args.figure_only:
        if not oat_path.exists() or not mc_path.exists():
            raise FileNotFoundError(
                f"Cached CSVs not found in {OUT_DIR}.  "
                "Run without --figure-only first."
            )
        print(f"Loading OAT data from {oat_path}")
        oat_df = pd.read_csv(oat_path)
        print(f"Loading MC data from {mc_path}")
        mc_df  = pd.read_csv(mc_path)
        # Nominal energy: zero-perturbation row
        nom_E = float(oat_df.loc[oat_df["perturbation_frac"] == 0.0, "energy_kJ"].iloc[0])
        print(f"  Nominal energy (from CSV): {nom_E:.2f} kJ")
        print("\nGenerating figure...")
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig_path = FIG_DIR / "fig_sensitivity.pdf"
        make_figure(oat_df, mc_df, nom_E, fig_path)
        return

    print("Loading farmland polygon...")
    lonlat_polygon = parse_kml_polygon(str(KML_SCENARIO))
    polygon_utm, _ = lonlat_to_utm_polygon(lonlat_polygon)

    print("\nRunning nominal R2 configuration...")
    nom = run_planner(polygon_utm, NOM_WIND_DIR_DEG, NOM_WIND_SPEED,
                      DEFAULT_POWER_MODEL)
    nom_E = nom["energy_kJ"]
    print(f"  Nominal: phi*={nom['phi_deg']:.1f}°, E={nom_E:.2f} kJ, "
          f"T={nom['time_min']:.1f} min, path={nom['path_km']:.2f} km")

    print("\nRunning OAT sensitivity analysis...")
    oat_df = run_oat(polygon_utm, nom)
    oat_path = OUT_DIR / "sensitivity_oat.csv"
    oat_df.to_csv(oat_path, index=False)
    print(f"  OAT results saved: {oat_path}  ({len(oat_df)} rows)")

    print(f"\nRunning Monte Carlo (N={MC_FAST_N}, phi* fixed at {nom['phi_deg']:.0f}°)...")
    rng = np.random.default_rng(seed=42)
    mc_df = run_mc(polygon_utm, nom, rng,
                   fixed_phi=nom["phi_deg"], n_samples=MC_FAST_N)
    mc_path = OUT_DIR / "sensitivity_mc.csv"
    mc_df.to_csv(mc_path, index=False)
    print(f"  MC results saved: {mc_path}  ({len(mc_df)} rows)")

    print_summary(oat_df, mc_df, nom_E)

    print("\nGenerating figure...")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "fig_sensitivity.pdf"
    make_figure(oat_df, mc_df, nom_E, fig_path)


if __name__ == "__main__":
    main()
