#!/usr/bin/env python3
"""
03_plot_results.py
==================
Parse the most recent PX4 SITL ULog, extract the GPS ground track and
airspeed profile, and overlay them on the farmland polygon.

Saves the figure to paper/figures/fig_sitl_validation.pdf and prints a
brief mission summary.

Usage (after running 02_upload_and_fly.py):

    source .venv/bin/activate
    python code/sitl_validation/03_plot_results.py [/path/to/log.ulg]

If no path is given the most recent .ulg under
  /home/yliu/PX4-Autopilot/build/px4_sitl_default/rootfs/fs/microsd/log/
is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from code/ and code/gcs_plugin/
_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "gcs_plugin"))

import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pyproj import Transformer
from pyulog import ULog

from gcs_plugin.coords import parse_kml_polygon, lonlat_to_utm_polygon

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT  = Path(__file__).resolve().parents[2]
KML_FILE   = REPO_ROOT / "code" / "scenarios" / "Farmland.kml"
OUT_PDF    = REPO_ROOT / "paper" / "figures" / "fig_sitl_validation.pdf"
LOG_ROOT   = Path(
    "/home/yliu/PX4-Autopilot/build/px4_sitl_default"
    "/rootfs/log"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_ulg(log_root: Path) -> Path:
    """Return the ULog most likely to contain a mission flight.

    Strategy: among all .ulg files, pick the one whose GPS track covers the
    largest latitude range.  An active mission flight moves ~400 m (≈0.004°),
    while an idle/parked log has near-zero extent.  Falls back to the
    alphabetically latest log if no log shows measurable movement.
    """
    logs = sorted(log_root.rglob("*.ulg"))
    if not logs:
        raise FileNotFoundError(
            f"No .ulg files found under {log_root}.\n"
            "Has SITL been run yet?  (run 01_run_sitl.sh + 02_upload_and_fly.py)"
        )

    best_log    = logs[-1]
    best_extent = 0.0

    for log in logs:
        try:
            u  = ULog(str(log))
            ds = {d.name: d for d in u.data_list}
            gps = ds.get("vehicle_gps_position") or ds.get("sensor_gps")
            if gps is None or len(gps.data.get("timestamp", [])) < 50:
                continue
            key = "latitude_deg" if "latitude_deg" in gps.data else "lat"
            lat = (gps.data[key] if key == "latitude_deg"
                   else gps.data[key] / 1e7)
            extent = float(lat.max() - lat.min())
            if extent > best_extent:
                best_extent = extent
                best_log    = log
        except Exception:
            pass

    return best_log


def load_gps_track(ulg: ULog) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lat (deg), lon (deg), timestamp (s) arrays from vehicle_gps_position.

    Trims to the armed interval so post-landing idle GPS is excluded.
    """
    datasets = {d.name: d for d in ulg.data_list}

    gps = datasets.get("vehicle_gps_position") or datasets.get("sensor_gps")
    if gps is None:
        raise KeyError(
            "No 'vehicle_gps_position' or 'sensor_gps' topic in log.\n"
            f"Available topics: {sorted(datasets)}"
        )

    # PX4 v1.14+ uses latitude_deg/longitude_deg (float, already in degrees)
    # Older versions used lat/lon as int32 × 1e-7
    if "latitude_deg" in gps.data:
        lat = gps.data["latitude_deg"]
        lon = gps.data["longitude_deg"]
    else:
        lat = gps.data["lat"] / 1e7
        lon = gps.data["lon"] / 1e7
    t_s = gps.data["timestamp"] / 1e6 # seconds

    # Remove rows with zero fix (lat == lon == 0)
    valid = (lat != 0.0) | (lon != 0.0)
    lat, lon, t_s = lat[valid], lon[valid], t_s[valid]

    # Trim to the LAST mission flight in the log.
    # Strategy: find the last time seq_reached reset to a low value (new mission
    # upload), use that as mission start.  Falls back to first arm event.
    # PX4 never disarms in SITL, so we cannot use actuator_armed for end.
    mission_ds = datasets.get("mission_result")
    if mission_ds is not None:
        mr_t   = mission_ds.data["timestamp"] / 1e6
        mr_seq = mission_ds.data["seq_reached"].astype(float)

        # Detect mission reset: a drop of >5 in seq_reached means a new
        # mission was uploaded and execution restarted from 0.
        seq_diffs = np.diff(mr_seq)
        reset_idxs = np.where(seq_diffs < -5)[0]
        if reset_idxs.size > 0:
            # Multiple missions: use last reset
            t_mission_start = mr_t[reset_idxs[-1] + 1]
        elif mr_seq[0] < 0:
            # Single mission starting from -1: find when execution began
            first_active = np.where(mr_seq >= 0)[0]
            t_mission_start = (mr_t[first_active[0]] if first_active.size > 0
                               else mr_t[0])
        else:
            # seq already positive at log start — use first arm event
            armed_ds = datasets.get("actuator_armed")
            if armed_ds is not None:
                t_arm_us = armed_ds.data["timestamp"]
                t_mission_start = (
                    t_arm_us[armed_ds.data["armed"].astype(bool)][0] / 1e6
                )
            else:
                t_mission_start = t_s[0]

        # Start trim at the mission start time (when seq_reached first fires),
        # not 30 s before it.  The 30 s pre-buffer captured the transit leg
        # from the home position to the first survey waypoint, producing a
        # spurious cross-field line and misplacing the start marker at home.
        t_start_trim = t_mission_start
        last_wp_t    = mr_t[mr_seq == mr_seq.max()][-1]
        # Trim at exactly last_wp_t so that the post-mission return-to-launch
        # (RTL) leg is excluded.  The +60 s buffer used previously included
        # the RTL segment, which reconnected the aircraft to the start position
        # and made the open tour appear as a closed loop.
        mask = (t_s >= t_start_trim) & (t_s <= last_wp_t)
        lat, lon, t_s = lat[mask], lon[mask], t_s[mask]

    return lat, lon, t_s


def load_airspeed(ulg: ULog, t_gps: np.ndarray) -> np.ndarray | None:
    """Return airspeed (m/s) interpolated to the GPS timestamps, or None."""
    datasets = {d.name: d for d in ulg.data_list}
    if "airspeed" not in datasets:
        return None
    asp = datasets["airspeed"]
    t_asp = asp.data["timestamp"] / 1e6
    v_asp = asp.data["true_airspeed_m_s"]
    # Interpolate onto GPS timestamps (clamp to available range)
    t_lo, t_hi = t_asp[0], t_asp[-1]
    t_clamp = np.clip(t_gps, t_lo, t_hi)
    return np.interp(t_clamp, t_asp, v_asp)


def lonlat_to_utm17n(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project WGS-84 lon/lat arrays to UTM zone 17N (EPSG:32617)."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
    easting, northing = t.transform(lon, lat)
    return np.asarray(easting), np.asarray(northing)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Find / load log ───────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = find_latest_ulg(LOG_ROOT)

    print(f"Parsing: {log_path}")
    ulg = ULog(str(log_path))

    # ── Extract GPS track ─────────────────────────────────────────────────────
    lat, lon, t_s = load_gps_track(ulg)
    airspeed = load_airspeed(ulg, t_s)

    duration_min = (t_s[-1] - t_s[0]) / 60.0
    print(f"GPS samples : {len(lat)}")
    print(f"Duration    : {duration_min:.2f} min")
    if airspeed is not None:
        print(f"Airspeed    : {airspeed.mean():.1f} m/s mean  "
              f"[{airspeed.min():.1f}–{airspeed.max():.1f}]")

    # ── Project to UTM 17N ────────────────────────────────────────────────────
    easting, northing = lonlat_to_utm17n(lon, lat)

    # ── Load farmland polygon ─────────────────────────────────────────────────
    poly_lonlat = parse_kml_polygon(KML_FILE)
    poly_utm, _crs = lonlat_to_utm_polygon(poly_lonlat)
    poly_x, poly_y = poly_utm.exterior.xy

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))

    # Polygon
    ax.fill(poly_x, poly_y, color="#d4edda", zorder=1)
    ax.plot(poly_x, poly_y, "k-", linewidth=1.2, zorder=2)

    # Ground track, coloured by airspeed (or uniform blue if no airspeed)
    if airspeed is not None:
        vmin, vmax = 10.0, 22.0   # m/s range for colourmap
        norm  = Normalize(vmin=vmin, vmax=vmax)
        cmap  = cm.plasma
        # Segment-by-segment colouring
        for i in range(len(easting) - 1):
            c = cmap(norm(float(airspeed[i])))
            ax.plot(easting[i:i+2], northing[i:i+2], color=c, linewidth=0.8, zorder=3)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Airspeed (m/s)", fontsize=9)
    else:
        ax.plot(easting, northing, color="#1f77b4", linewidth=0.8, zorder=3)

    # Start / end markers
    ax.plot(easting[0],  northing[0],  "g^", markersize=8, label="Start", zorder=4)
    ax.plot(easting[-1], northing[-1], "rs", markersize=8, label="End",   zorder=4)

    ax.set_aspect("equal")
    ax.set_xlabel("Easting — UTM 17N (m)", fontsize=10)
    ax.set_ylabel("Northing — UTM 17N (m)", fontsize=10)
    ax.set_title(
        f"PX4 SITL ground track — farmland R2 baseline\n"
        f"Duration {duration_min:.2f} min  (model: 7.5 min)",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=150)
    print(f"\nFigure saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
