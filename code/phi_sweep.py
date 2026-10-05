"""
phi_sweep.py
============
T(phi) orientation sweep and optimal-phi analysis.

For each scenario polygon x wind ratio rho, sweeps orientation phi in [0, 179]
in 1-degree steps and records estimated flight time T(phi).  Also implements a
golden-section search for phi* = argmin T(phi) in ~20 planner calls.

Produces:
  paper/figures/fig_phi_landscape.pdf  -- 4-panel T(phi) for all scenarios at rho=0.2
  paper/figures/fig_phi_crossover.pdf  -- corridor scenario: T(phi) at rho=0.1,0.2,0.3
  paper/data/phi_sweep_results.csv     -- full sweep data
  paper/data/phi_sweep_summary.txt     -- per-scenario geometry + crossover summary

Usage
-----
    python3 code/phi_sweep.py                           # all scenarios, rho in {0.1,0.2,0.3}
    python3 code/phi_sweep.py --rho 0.1 0.2 0.3        # explicit rho values
    python3 code/phi_sweep.py --cell-size 20            # cell size in metres (default 20)
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from instance_gen import load_scenario_polygon
from decomposition import decompose
from routing import build_route, PowerModel, DEFAULT_POWER_MODEL, energy_cost

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_DIR    = Path(__file__).parent
SCENARIOS_DIR = CODE_DIR / "scenarios"
FIGURES_DIR = CODE_DIR.parent / "paper" / "figures"
DATA_DIR    = CODE_DIR.parent / "paper" / "data"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

AIRSPEED_MPS = 12.0

# Scenario registry: JSON filename -> display name
SCENARIO_FILES = [
    ("scenario_01.json",                   "Concave Farmland"),
    ("scenario_convex_meadow.json",        "Convex Meadow"),
    ("scenario_high_concave_terrain.json", "High-Concave Terrain"),
    ("scenario_corridor_channel.json",     "Corridor Channel"),
]

# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def _wind_vec(wind_dir_deg: float, wind_ratio: float, airspeed: float) -> np.ndarray:
    """Wind vector from geographic direction and speed ratio."""
    wind_speed = wind_ratio * airspeed
    rad = math.radians(wind_dir_deg)
    return np.array([wind_speed * math.cos(rad), wind_speed * math.sin(rad)])


def sweep_phi(polygon, wind_dir_deg: float, wind_ratio: float,
              airspeed: float = AIRSPEED_MPS,
              cell_size: float = 20.0,
              phi_step: int = 1) -> pd.DataFrame:
    """
    Sweep grid orientation phi in [0, 179] degrees (phi_step increments).
    For each phi, run decompose() + build_route() and record T(phi).

    Returns a DataFrame with columns:
        phi_deg, flight_time_s, n_cells, n_strips
    """
    wind_vec = _wind_vec(wind_dir_deg, wind_ratio, airspeed)
    rows = []
    for phi in range(0, 180, phi_step):
        decomp = decompose(polygon, cell_size, cell_size, float(phi))
        route  = build_route(decomp, wind_vec=wind_vec, airspeed=airspeed)
        rows.append({
            "phi_deg":       phi,
            "flight_time_s": route.total_time_s,
            "n_cells":       decomp.n_cells,
            "n_strips":      route.metrics["n_strips"],
        })
    return pd.DataFrame(rows)


def sweep_phi_energy(polygon, wind_dir_deg: float, wind_ratio: float,
                     airspeed: float = AIRSPEED_MPS,
                     cell_size: float = 20.0,
                     phi_step: int = 1,
                     power_model: PowerModel = DEFAULT_POWER_MODEL) -> pd.DataFrame:
    """
    Sweep grid orientation phi in [0, 179] degrees under the energy-optimal objective.

    Strip direction choice and strip ordering minimise total energy (not time).
    Per-leg airspeed v_a*(theta) is chosen optimally via ``optimal_airspeed()``.

    Returns a DataFrame with columns:
        phi_deg, total_energy_J, flight_time_s, n_cells, n_strips, va_mean_ms
    """
    wind_vec = _wind_vec(wind_dir_deg, wind_ratio, airspeed)
    rows = []
    for phi in range(0, 180, phi_step):
        decomp = decompose(polygon, cell_size, cell_size, float(phi))
        route  = build_route(decomp, wind_vec=wind_vec, airspeed=airspeed,
                             power_model=power_model, objective="energy")
        mean_va = (sum(route.leg_airspeeds_mps) / len(route.leg_airspeeds_mps)
                   if route.leg_airspeeds_mps else airspeed)
        rows.append({
            "phi_deg":        phi,
            "total_energy_J": route.total_energy_J,
            "flight_time_s":  route.total_time_s,
            "n_cells":        decomp.n_cells,
            "n_strips":       route.metrics["n_strips"],
            "va_mean_mps":    mean_va,
        })
    return pd.DataFrame(rows)


def sweep_phi_dual(polygon, wind_dir_deg: float, wind_ratio: float,
                   airspeed: float = AIRSPEED_MPS,
                   cell_size: float = 20.0,
                   phi_step: int = 1,
                   power_model: PowerModel = DEFAULT_POWER_MODEL) -> pd.DataFrame:
    """
    Combined sweep computing both T(phi) (time-optimal routing at fixed v_a)
    and E(phi) (energy-optimal routing with per-leg v_a*) in one pass.

    Returns a DataFrame with columns:
        phi_deg, time_opt_T_s, time_opt_E_J, energy_opt_T_s, energy_opt_E_J,
        n_cells, n_strips, va_mean_energy_mps
    """
    wind_vec = _wind_vec(wind_dir_deg, wind_ratio, airspeed)
    rows = []
    for phi in range(0, 180, phi_step):
        decomp = decompose(polygon, cell_size, cell_size, float(phi))
        rt = build_route(decomp, wind_vec=wind_vec, airspeed=airspeed,
                         power_model=power_model, objective="time")
        re = build_route(decomp, wind_vec=wind_vec, airspeed=airspeed,
                         power_model=power_model, objective="energy")
        mean_va_e = (sum(re.leg_airspeeds_mps) / len(re.leg_airspeeds_mps)
                     if re.leg_airspeeds_mps else airspeed)
        rows.append({
            "phi_deg":            phi,
            "time_opt_T_s":       rt.total_time_s,
            "time_opt_E_J":       rt.total_energy_J,
            "energy_opt_T_s":     re.total_time_s,
            "energy_opt_E_J":     re.total_energy_J,
            "n_cells":            decomp.n_cells,
            "n_strips":           rt.metrics["n_strips"],
            "va_mean_energy_mps": mean_va_e,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Golden-section search for phi*
# ---------------------------------------------------------------------------

def find_optimal_phi(polygon, wind_vec: np.ndarray,
                     airspeed: float = AIRSPEED_MPS,
                     cell_size: float = 20.0,
                     tol: float = 0.5) -> tuple[float, float, int]:
    """
    Find phi* = argmin_{phi in [0, 179]} T(phi) using golden-section search.

    T(phi) may be multi-modal for corridor / high-concave polygons.
    When that occurs, golden-section returns a *local* minimum.
    A 4-start variant is used: initial seeds at 0, 45, 90, 135 degrees;
    the global best local minimum is returned.

    Returns
    -------
    phi_star : float  -- optimal orientation (degrees)
    t_star   : float  -- T(phi_star) in seconds
    n_calls  : int    -- total number of T evaluations
    """
    GOLDEN_RATIO = (math.sqrt(5) - 1) / 2

    call_count = [0]

    def T(phi_float: float) -> float:
        phi = float(phi_float)
        call_count[0] += 1
        decomp = decompose(polygon, cell_size, cell_size, phi)
        route  = build_route(decomp, wind_vec=wind_vec, airspeed=airspeed)
        return route.total_time_s

    def gss(a: float, b: float, tol: float) -> tuple[float, float]:
        """Golden-section search on interval [a, b]."""
        c = b - GOLDEN_RATIO * (b - a)
        d = a + GOLDEN_RATIO * (b - a)
        fc, fd = T(c), T(d)
        while (b - a) > tol:
            if fc < fd:
                b = d
                d, fd = c, fc
                c = b - GOLDEN_RATIO * (b - a)
                fc = T(c)
            else:
                a = c
                c, fc = d, fd
                d = a + GOLDEN_RATIO * (b - a)
                fd = T(d)
        phi_min = (a + b) / 2
        return phi_min, T(phi_min)

    # 4-start: divide [0,180] into four sub-intervals
    starts = [(0, 44), (45, 89), (90, 134), (135, 179)]
    best_phi, best_t = None, math.inf
    for a, b in starts:
        phi_loc, t_loc = gss(float(a), float(b), tol)
        if t_loc < best_t:
            best_t = t_loc
            best_phi = phi_loc

    return best_phi, best_t, call_count[0]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_PALETTE = {
    0.1: "#2c7bb6",
    0.2: "#d7191c",
    0.3: "#1a9641",
}


def figure_phi_landscape(all_sweep_data: dict, out_path: Path = None) -> Path:
    """
    2x2 panel: one scenario per axis.  Shows both T(phi) (time-optimal, solid)
    and E(phi) (energy-optimal, dashed), normalised to [0,1] for visual
    comparison, at rho=0.2.  Marks phi_T* (red solid) and phi_E* (blue dashed)
    vertical lines, and phi_wind (green dotted).

    all_sweep_data: dict mapping scenario_name -> {rho: dual-sweep DataFrame}
    """
    scenario_names = list(all_sweep_data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    axes = axes.flatten()

    RHO_SHOW = 0.2

    for ax, name in zip(axes, scenario_names):
        if RHO_SHOW not in all_sweep_data[name]:
            ax.set_visible(False)
            continue
        df = all_sweep_data[name][RHO_SHOW]

        # Normalise both curves to [0, 1] for visual overlay
        T_vals = df["time_opt_T_s"].values
        E_vals = df["energy_opt_E_J"].values
        T_norm = (T_vals - T_vals.min()) / (T_vals.max() - T_vals.min() + 1e-12)
        E_norm = (E_vals - E_vals.min()) / (E_vals.max() - E_vals.min() + 1e-12)

        ax.plot(df["phi_deg"], T_norm, color="#d7191c", linewidth=1.4,
                label=r"$T(\phi)$ (time-opt)")
        ax.plot(df["phi_deg"], E_norm, color="#2c7bb6", linewidth=1.4,
                linestyle="--", label=r"$E(\phi)$ (energy-opt)")

        # phi_T*, phi_E*, phi_wind
        phi_T_star = df.loc[df["time_opt_T_s"].idxmin(), "phi_deg"]
        phi_E_star = df.loc[df["energy_opt_E_J"].idxmin(), "phi_deg"]
        # get wind dir from SCENARIO_INFO if available, else from the dataframe
        if name in SCENARIO_INFO:
            phi_wind = SCENARIO_INFO[name]["wind_dir_deg"]
        elif "wind_dir_deg" in df.columns:
            phi_wind = df["wind_dir_deg"].iloc[0]
        else:
            phi_wind = None

        ax.axvline(phi_T_star, color="#d7191c", linewidth=1.3,
                   linestyle="-",  label=f"$\\phi_T^*={phi_T_star:.0f}^\\circ$")
        ax.axvline(phi_E_star, color="#2c7bb6", linewidth=1.3,
                   linestyle="--", label=f"$\\phi_E^*={phi_E_star:.0f}^\\circ$")
        if phi_wind is not None:
            ax.axvline(phi_wind % 180, color="#1a9641", linewidth=1.2,
                       linestyle=":",  label=f"$\\phi_{{\\mathrm{{wind}}}}={phi_wind:.0f}^\\circ$")

        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Grid orientation $\\phi$ (\u00b0)", fontsize=13)
        ax.set_ylabel("Normalised objective (a.u.)", fontsize=13)
        ax.set_xlim(0, 179)
        ax.set_ylim(-0.05, 1.15)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=11, loc="upper right", ncol=2)
        ax.grid(True, linestyle="--", alpha=0.35)

    plt.suptitle(
        "Orientation landscapes $T(\\phi)$ and $E(\\phi)$ for all four benchmark scenarios\n"
        "($\\rho = 0.2$, $s = 20\\,\\mathrm{m}$; curves normalised to $[0,1]$; "
        "red = time-opt, blue = energy-opt, green = wind direction)",
        fontsize=12, y=1.01)
    plt.tight_layout(pad=0.8)

    out = out_path or (FIGURES_DIR / "fig_phi_landscape.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def figure_phi_crossover(scenario_data: dict, phi_wind_deg: float,
                          out_path: Path = None) -> Path:
    """
    Single panel: Corridor Channel scenario E(phi) (energy-optimal) at multiple
    rho values.  Shows that corridor geometry forces phi*=90 regardless of wind ratio.
    scenario_data: {rho: dual-sweep DataFrame}
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    rhos = sorted(scenario_data.keys())
    colors = [_PALETTE.get(r, "#888888") for r in rhos]

    for rho, color in zip(rhos, colors):
        df = scenario_data[rho]
        phi_E_star = df.loc[df["energy_opt_E_J"].idxmin(), "phi_deg"]
        ax.plot(df["phi_deg"], df["energy_opt_E_J"] / 1000.0,
                color=color, linewidth=1.5,
                label=f"$E(\\phi)$, $\\rho={rho}$ ($\\phi^*={phi_E_star:.0f}^\\circ$)")

    ax.axvline(phi_wind_deg % 180, color="#555555", linewidth=1.3,
               linestyle="--", label=f"$\\phi_{{\\mathrm{{wind}}}}={phi_wind_deg:.0f}^\\circ$")

    ax.set_xlabel("Grid orientation $\\phi$ (°)", fontsize=13)
    ax.set_ylabel("Est. mission energy $E(\\phi)$ (kJ)", fontsize=13)
    ax.set_title("Corridor Channel: $E(\\phi)$ at multiple wind ratios\n"
                 "(corridor geometry forces $\\phi^*=90^\\circ$ regardless of wind ratio)",
                 fontsize=12)
    ax.set_xlim(0, 179)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout(pad=0.8)

    out = out_path or (FIGURES_DIR / "fig_phi_crossover.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Geometry summary (for overshoot bound)
# ---------------------------------------------------------------------------

def polygon_geometry_summary(polygon, name: str) -> dict:
    """Return area, perimeter, and overshoot bound for one scenario polygon."""
    area      = polygon.area
    perimeter = polygon.exterior.length
    w = h = 20.0  # default cell size
    n_boundary_bound = perimeter / min(w, h)
    overshoot_bound  = (perimeter * min(w, h)) / (2.0 * area)  # fraction
    return {
        "scenario":          name,
        "area_m2":           area,
        "perimeter_m":       perimeter,
        "overshoot_bound":   overshoot_bound,
        "overshoot_bound_pct": 100.0 * overshoot_bound,
        "iso_ratio":         2.0 * area / perimeter,  # r_iso = 2A/l
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Will be populated in main() and used by figure functions
SCENARIO_INFO: dict = {}


def main():
    parser = argparse.ArgumentParser(
        description="Sweep T(phi) for all scenarios and produce figures")
    parser.add_argument("--rho", nargs="+", type=float, default=[0.1, 0.2, 0.3],
                        help="Wind ratios (default: 0.1 0.2 0.3)")
    parser.add_argument("--cell-size", type=float, default=20.0,
                        help="Cell size in metres (default: 20)")
    parser.add_argument("--phi-step", type=int, default=1,
                        help="Phi sweep step in degrees (default: 1)")
    args = parser.parse_args()

    cell_size = args.cell_size
    rhos      = args.rho
    phi_step  = args.phi_step
    airspeed  = AIRSPEED_MPS

    all_rows = []
    all_sweep_data = {}   # {scenario_name: {rho: DataFrame}}
    geometry_rows  = []

    print(f"Sweeping phi in [0, 179] at {phi_step}° steps, "
          f"cell_size={cell_size}m, rho={rhos}, airspeed={airspeed}m/s\n")

    for json_file, display_name in SCENARIO_FILES:
        scenario_path = SCENARIOS_DIR / json_file
        if not scenario_path.exists():
            print(f"  [skip] {json_file} not found")
            continue

        with open(scenario_path) as f:
            cfg = json.load(f)

        polygon      = load_scenario_polygon(str(scenario_path))
        wind_dir_deg = cfg["wind"]["direction_deg"]

        # Store in global for figure functions
        SCENARIO_INFO[display_name] = {
            "wind_dir_deg": wind_dir_deg,
            "polygon":      polygon,
        }

        # Geometry summary
        geom = polygon_geometry_summary(polygon, display_name)
        geometry_rows.append(geom)
        print(f"{display_name}")
        print(f"  Area       = {geom['area_m2']:,.0f} m²")
        print(f"  Perimeter  = {geom['perimeter_m']:,.0f} m")
        print(f"  Overshoot bound (w=h=20m): {geom['overshoot_bound_pct']:.1f}%")
        print(f"  r_iso      = {geom['iso_ratio']:.1f} m")

        all_sweep_data[display_name] = {}

        for rho in rhos:
            print(f"  rho={rho}  phi dual sweep ... ", end="", flush=True)
            df = sweep_phi_dual(polygon, wind_dir_deg, rho, airspeed, cell_size, phi_step)
            all_sweep_data[display_name][rho] = df

            phi_T_star = int(df.loc[df["time_opt_T_s"].idxmin(), "phi_deg"])
            phi_E_star = int(df.loc[df["energy_opt_E_J"].idxmin(), "phi_deg"])
            t_star     = df["time_opt_T_s"].min()

            print(f"phi_T*={phi_T_star}° (T*={t_star:.1f}s)  "
                  f"phi_E*={phi_E_star}°  divergence={abs(phi_T_star - phi_E_star)}°")

            # Accumulate CSV rows
            for _, row in df.iterrows():
                all_rows.append({
                    "scenario":        display_name,
                    "wind_dir_deg":    wind_dir_deg,
                    "rho":             rho,
                    "phi_deg":         int(row["phi_deg"]),
                    "time_opt_T_s":    row["time_opt_T_s"],
                    "time_opt_E_J":    row["time_opt_E_J"],
                    "energy_opt_T_s":  row["energy_opt_T_s"],
                    "energy_opt_E_J":  row["energy_opt_E_J"],
                    "n_cells":         int(row["n_cells"]),
                    "n_strips":        int(row["n_strips"]),
                })

        print()

    # Golden-section benchmark on corridor scenario
    print("--- Golden-section benchmark (Corridor Channel, rho=0.2) ---")
    corridor_json = SCENARIOS_DIR / "scenario_corridor_channel.json"
    if corridor_json.exists():
        with open(corridor_json) as f:
            cfg_c = json.load(f)
        poly_c   = load_scenario_polygon(str(corridor_json))
        wdir_c   = cfg_c["wind"]["direction_deg"]
        wvec_c   = _wind_vec(wdir_c, 0.2, airspeed)
        phi_gs, t_gs, n_calls = find_optimal_phi(poly_c, wvec_c, airspeed, cell_size)
        print(f"  phi* = {phi_gs:.1f}°,  T* = {t_gs:.1f} s,  evaluations = {n_calls}")
        print(f"  (full 1°-sweep uses 180 evaluations)")

    # Save CSV
    csv_path = DATA_DIR / "phi_sweep_results.csv"
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}  ({len(df_all)} rows)")

    # Save geometry summary
    geom_df = pd.DataFrame(geometry_rows)
    geom_path = DATA_DIR / "phi_sweep_geometry.csv"
    geom_df.to_csv(geom_path, index=False)
    print(f"Saved: {geom_path}")

    # Save human-readable summary
    summary_path = DATA_DIR / "phi_sweep_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Scenario geometry and overshoot bounds (cell size w=h=20m)\n")
        f.write("=" * 62 + "\n")
        for g in geometry_rows:
            f.write(f"\n{g['scenario']}\n")
            f.write(f"  Area       = {g['area_m2']:>10,.0f} m²\n")
            f.write(f"  Perimeter  = {g['perimeter_m']:>10,.0f} m\n")
            f.write(f"  r_iso      = {g['iso_ratio']:>10.1f} m\n")
            f.write(f"  Overshoot bound = {g['overshoot_bound_pct']:.2f}%\n")
        f.write("\nDual-objective orientation summary (phi_T* vs phi_E*)\n")
        f.write("-" * 62 + "\n")
        for sc_name, sc_data in all_sweep_data.items():
            wind_deg = SCENARIO_INFO[sc_name]["wind_dir_deg"]
            f.write(f"\n{sc_name}  (phi_wind={wind_deg:.0f}°)\n")
            for rho in sorted(sc_data.keys()):
                df = sc_data[rho]
                phi_T_star = int(df.loc[df["time_opt_T_s"].idxmin(), "phi_deg"])
                phi_E_star = int(df.loc[df["energy_opt_E_J"].idxmin(), "phi_deg"])
                t_star     = df["time_opt_T_s"].min()
                e_star     = df["energy_opt_E_J"].min()
                f.write(f"  rho={rho}:  phi_T*={phi_T_star:3d}° T*={t_star:.1f}s  "
                        f"phi_E*={phi_E_star:3d}° E*={e_star/1000:.2f}kJ  "
                        f"diverge={abs(phi_T_star - phi_E_star)}°\n")
    print(f"Saved: {summary_path}")

    # Produce figures
    print("\nGenerating figures ...")
    figure_phi_landscape(all_sweep_data)
    if "Corridor Channel" in all_sweep_data:
        corridor_wind = SCENARIO_INFO["Corridor Channel"]["wind_dir_deg"]
        figure_phi_crossover(all_sweep_data["Corridor Channel"], corridor_wind)

    print("\nDone.")


if __name__ == "__main__":
    main()
