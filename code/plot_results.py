"""
plot_results.py
===============
Produce all manuscript figures from saved experiment data.

Reads paper/data/results_{tag}.csv and calls visualise helpers to produce
Figures 1–4 as specified in sec_experiments.tex.

Usage
-----
    python plot_results.py --tag pilot       # uses results_pilot.csv
    python plot_results.py --tag full        # uses results_full.csv
    python plot_results.py --wind-sweep      # re-generate fig_wind_sweep.pdf
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from visualize import (
    figure_wind_sweep,
    figure_cell_count,
    figure_runtime,
    FIGURES_DIR,
)

DATA_DIR = Path(__file__).parent.parent / "paper" / "data"
TABLES_DIR = Path(__file__).parent.parent / "paper" / "tables"


def plot_cell_count_from_df(df: pd.DataFrame) -> None:
    """Figure 3: Retained-cell count vs. concavity/cell size."""
    sub = (df[df["method"] == "proposed"]
           .groupby(["family", "cell_w"], as_index=False)
           .agg(n_cells=("n_cells", "mean")))
    sub = sub.rename(columns={"cell_w": "cell_size"})
    figure_cell_count(sub)


def plot_runtime_from_df(df: pd.DataFrame) -> None:
    """Figure 4: Planner runtime vs. problem size."""
    sub = df[df["method"] == "proposed"][["n_cells", "runtime_s"]].copy()
    figure_runtime(sub)


def plot_wind_sweep_from_df(df: pd.DataFrame, family: str = "mild_concave",
                              wind_ratio: float = 0.25) -> None:
    """Figure 2: Estimated flight time vs. wind direction."""
    # Use a fixed wind speed close to 0.25 ratio
    ratio_vals = sorted(df["wind_ratio"].unique())
    chosen_ratio = min(ratio_vals, key=lambda r: abs(r - wind_ratio))

    proposed = (df[(df["method"] == "proposed") &
                   (df["family"] == family) &
                   (df["wind_ratio"].round(3) == round(chosen_ratio, 3))]
                .groupby("wind_deg")["flight_time_s"].mean())
    baseline = (df[(df["method"] == "baseline1") &
                   (df["family"] == family) &
                   (df["wind_ratio"].round(3) == round(chosen_ratio, 3))]
                .groupby("wind_deg")["flight_time_s"].mean())

    nowind = (df[(df["method"] == "proposed") &
                 (df["family"] == family) &
                 (df["wind_ratio"] == 0.0)]
              ["flight_time_s"].mean())

    dirs = sorted(set(proposed.index) & set(baseline.index))
    if not dirs:
        print(f"[warn] No matching rows for wind sweep (family={family}, ratio={chosen_ratio})")
        return

    figure_wind_sweep(
        wind_dirs_deg=dirs,
        proposed_times=[proposed[d] for d in dirs],
        baseline_times=[baseline[d] for d in dirs],
        nowind_time=nowind,
    )


def main():
    parser = argparse.ArgumentParser(description="Produce manuscript figures from results CSV")
    parser.add_argument("--tag", default="pilot",
                         help="Results file tag: pilot or full (default: pilot)")
    parser.add_argument("--family", default="mild_concave",
                         help="Polygon family for wind-sweep figure (default: mild_concave)")
    args = parser.parse_args()

    csv_path = DATA_DIR / f"results_{args.tag}.csv"
    if not csv_path.exists():
        print(f"Error: results file not found: {csv_path}")
        print("Run experiments.py first.")
        sys.exit(1)

    print(f"Loading results from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, methods: {df['method'].unique().tolist()}")

    print("Generating Figure 3: cell count...")
    plot_cell_count_from_df(df)

    print("Generating Figure 4: runtime...")
    plot_runtime_from_df(df)

    print(f"Generating Figure 2: wind sweep (family={args.family})...")
    plot_wind_sweep_from_df(df, family=args.family)

    print("\nAll figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
