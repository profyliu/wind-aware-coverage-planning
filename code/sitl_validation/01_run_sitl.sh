#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 01_run_sitl.sh
# Launch PX4 SITL + jMAVSim (headless) at the SE Michigan farmland site.
#
# Prerequisites:
#   - PX4-Autopilot already built at /home/yliu/PX4-Autopilot
#     (run: cd /home/yliu/PX4-Autopilot && make px4_sitl_default)
#   - Java 11+ installed (jMAVSim requires it)
#   - ant installed  (jMAVSim build tool)
#
# After SITL is running (you see "INFO [commander] Ready for takeoff"),
# open a second terminal and run:
#   source .venv/bin/activate
#   python code/sitl_validation/02_upload_and_fly.py
#
# QGroundControl (Windows) can monitor the flight:
#   Add a UDP link to  <WSL2_IP>:18570  (IP printed below at startup).
#   If QGC cannot reach WSL2, run this in Windows PowerShell (as Admin):
#     $wsl = (wsl -e ip -4 addr show eth0) -match "inet\s+\d+" | ...
#     netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 `
#           listenport=18570 connectaddress=<WSL2_IP> connectport=18570
# ---------------------------------------------------------------------------

set -e

PX4_DIR=/home/yliu/PX4-Autopilot

# ── Install PX4 Python build dependencies if missing ─────────────────────────
# kconfiglib, empy, etc. must be on the *system* python3 (used by cmake/make).
python3 -c "import kconfiglib" 2>/dev/null || {
    echo "Installing PX4 build dependencies (kconfiglib, empy, …)…"
    pip3 install kconfiglib "empy>=3.3,<4" pyros-genmsg pyyaml toml future lark
}

# ── GPS origin: SE Michigan farmland centroid ─────────────────────────────────
export PX4_HOME_LAT=41.980759
export PX4_HOME_LON=-83.346720
export PX4_HOME_ALT=250          # approximate MSL altitude (m)

# ── Headless mode (no GUI window) ─────────────────────────────────────────────
export HEADLESS=1
# Force Java to headless mode in case AWT is initialised indirectly
export JAVA_TOOL_OPTIONS="-Djava.awt.headless=true"
# Speed up simulation (e.g. 3× faster = ~2.5 min instead of 7.5 min).
# Set to 1 for real-time, higher values for faster validation.
export PX4_SIM_SPEED_FACTOR=${PX4_SIM_SPEED_FACTOR:-3}

# ── Print connection info ─────────────────────────────────────────────────────
WSL_IP=$(ip -4 addr show eth0 2>/dev/null \
         | grep -oP '(?<=inet\s)\d+(\.\d+){3}' \
         | head -1 || echo "127.0.0.1")

echo "========================================================"
echo " PX4 SITL — farmland R2 baseline flyability validation"
echo " jMAVSim GPS origin: ${PX4_HOME_LAT}, ${PX4_HOME_LON}, ${PX4_HOME_ALT} m MSL"
echo ""
echo " MAVLink ports (after SITL starts):"
echo "   Python upload script  : udp:127.0.0.1:14540 (companion port)"
echo "   QGroundControl (Windows): ${WSL_IP}:18570"
echo ""
echo " Next step (second terminal):"
echo "   source .venv/bin/activate"
echo "   python code/sitl_validation/02_upload_and_fly.py"
echo "========================================================"
echo ""

# ── Launch SITL ───────────────────────────────────────────────────────────────
cd "$PX4_DIR"
make px4_sitl_default jmavsim
