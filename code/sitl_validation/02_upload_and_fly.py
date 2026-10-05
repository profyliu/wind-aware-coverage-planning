#!/usr/bin/env python3
"""
02_upload_and_fly.py
====================
Connect to a running PX4 SITL instance, upload the farmland R2 baseline
mission, arm the vehicle, and monitor progress until the mission completes.

Run *after* 01_run_sitl.sh reports "Ready for takeoff":

    source .venv/bin/activate
    python code/sitl_validation/02_upload_and_fly.py

Output printed to stdout; the PX4 ULog is written automatically by SITL
to /home/yliu/PX4-Autopilot/build/px4_sitl_default/rootfs/fs/microsd/log/.
Run 03_plot_results.py once the mission finishes to generate the figure.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from pymavlink import mavutil, mavwp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WAYPOINTS_FILE = (
    Path(__file__).resolve().parents[2]
    / "paper" / "data" / "farmland_paths" / "farmland_alt80_ov70_baseline.waypoints"
)

# PX4 SITL companion port: PX4 sends FROM 14580 TO 14540.
# We listen on 14540; pymavlink discovers PX4's address automatically.
SITL_ADDR = "udpin:0.0.0.0:14540"

# PX4 custom mode values (from px4_custom_mode.h)
PX4_MAIN_MODE_AUTO    = 4
PX4_SUB_MODE_MISSION  = 4  # PX4_CUSTOM_SUB_MODE_AUTO_MISSION

CUSTOM_MODE_MISSION = (PX4_SUB_MODE_MISSION << 24) | (PX4_MAIN_MODE_AUTO << 16)

# MAVLink frame types
MAV_FRAME_GLOBAL_RELATIVE_ALT = 3
MAV_FRAME_MISSION             = 2  # used for DO/action commands (no GPS position)

# PX4 SITL force-arm magic (bypasses pre-arm health checks)
PX4_ARM_FORCE_MAGIC = 21196

HEARTBEAT_TIMEOUT   = 60   # seconds to wait for PX4 heartbeat
UPLOAD_TIMEOUT      = 10   # seconds per MAVLink request during upload
ARM_TIMEOUT         = 30   # seconds to wait for arm confirmation
READY_TIMEOUT       = 30   # seconds to wait for STANDBY status before arming
MISSION_TIMEOUT     = 900  # 15 min max for the full mission


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wait_heartbeat(master: mavutil.mavudp, timeout: float) -> None:
    """Block until the first heartbeat arrives or raise TimeoutError."""
    print(f"  Waiting for PX4 heartbeat (timeout {timeout} s)…")
    master.wait_heartbeat(timeout=timeout)
    print(f"  Heartbeat from sysid={master.target_system}  "
          f"compid={master.target_component}")


def wait_for_standby(master: mavutil.mavudp, timeout: float) -> None:
    """Wait until PX4 reports MAV_STATE_STANDBY (pre-arm checks passed)."""
    print(f"  Waiting for STANDBY status (timeout {timeout} s)…")
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = master.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if msg is None:
            continue
        # MAV_STATE_STANDBY = 3
        if msg.system_status == mavutil.mavlink.MAV_STATE_STANDBY:
            print("  System STANDBY — pre-arm checks passed.")
            return
    # Don't fail hard; PX4 SITL sometimes skips STANDBY for ACTIVE
    print("  [warn] Did not reach STANDBY within timeout; attempting arm anyway.")


def set_mission_mode(master: mavutil.mavudp) -> None:
    """Send SET_MODE to switch PX4 into AUTO/MISSION."""
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        CUSTOM_MODE_MISSION,
    )
    # Wait for COMMAND_ACK or HEARTBEAT confirming the mode change
    deadline = time.time() + 10
    while time.time() < deadline:
        msg = master.recv_match(type=["COMMAND_ACK", "HEARTBEAT"],
                                blocking=True, timeout=2)
        if msg is None:
            continue
        if msg.get_type() == "HEARTBEAT":
            if msg.custom_mode == CUSTOM_MODE_MISSION:
                return
        elif msg.get_type() == "COMMAND_ACK":
            if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                return
    # Mode may still be set even without explicit ACK — proceed anyway
    print("  [warn] No explicit mode-change ACK; proceeding.")


def arm(master: mavutil.mavudp, timeout: float) -> None:
    """Send ARM command; fall back to force-arm if pre-arm checks block it."""
    for force in (False, True):
        param2 = PX4_ARM_FORCE_MAGIC if force else 0
        if force:
            print("  Normal arm rejected — retrying with force-arm (SITL bypass)…")
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,          # confirmation
            1,          # param1: 1 = arm
            float(param2), 0, 0, 0, 0, 0,
        )
        deadline = time.time() + (timeout if not force else 10)
        while time.time() < deadline:
            msg = master.recv_match(
                type=["COMMAND_ACK", "STATUSTEXT"],
                blocking=True, timeout=2,
            )
            if msg is None:
                continue
            if msg.get_type() == "COMMAND_ACK":
                if msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                    if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                        return
                    elif force:
                        raise RuntimeError(
                            f"Force-arm also rejected (result={msg.result}). "
                            "Check SITL console for health errors."
                        )
                    else:
                        break  # try force arm
            elif msg.get_type() == "STATUSTEXT":
                text = getattr(msg, "text", "").strip()
                if text and "arm" in text.lower():
                    print(f"  STATUSTEXT: {text}")
    raise TimeoutError(f"Vehicle did not arm within {timeout} s.")


def set_param(master: mavutil.mavudp, name: str, value: float,
              timeout: float = 5.0) -> None:
    """Set a PX4 REAL32 parameter via PARAM_SET and wait for PARAM_VALUE ACK."""
    name_bytes = name.encode("utf-8")[:16].ljust(16, b"\x00")
    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        name_bytes,
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=2)
        if msg is None:
            continue
        if msg.param_id.rstrip("\x00") == name:
            print(f"    {name:<24s} = {msg.param_value:.2f}")
            return
    print(f"  [warn] No PARAM_VALUE ACK for {name} within {timeout:.0f} s")


def configure_airframe(master: mavutil.mavudp,
                       cruise_ms: float = 21.5,
                       vel_max_ms: float = 25.0) -> None:
    """Raise PX4 velocity limits so DO_CHANGE_SPEED commands at ~21 m/s are honoured.

    MPC_XY_VEL_MAX  — hard ceiling on the horizontal velocity setpoint (default 12 m/s).
    MPC_XY_CRUISE   — default mission cruise speed used when no DO_CHANGE_SPEED is
                      present (does not override DO_CHANGE_SPEED, but must be <= VEL_MAX).
    """
    print("Configuring airframe velocity parameters …")
    set_param(master, "MPC_XY_VEL_MAX", vel_max_ms)
    set_param(master, "MPC_XY_CRUISE",  cruise_ms)


def upload_mission(master: mavutil.mavudp, wp_loader: mavwp.MAVWPLoader) -> int:
    """Upload mission items; return the count of NAV_WAYPOINT items uploaded."""
    target_sys  = master.target_system
    target_comp = master.target_component
    count       = wp_loader.count()

    # Clear any existing mission
    master.mav.mission_clear_all_send(target_sys, target_comp,
                                      mavutil.mavlink.MAV_MISSION_TYPE_MISSION)
    ack = master.recv_match(type="MISSION_ACK", blocking=True, timeout=UPLOAD_TIMEOUT)
    if ack is None:
        print("  [warn] No MISSION_ACK after clear; proceeding anyway.")

    # Send item count
    master.mav.mission_count_send(target_sys, target_comp, count,
                                  mavutil.mavlink.MAV_MISSION_TYPE_MISSION)

    # MAVLink commands that carry GPS positions vs. DO/action commands
    _NAV_CMDS = {
        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM,
        mavutil.mavlink.MAV_CMD_NAV_LOITER_TIME,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
    }

    # Respond to each MISSION_REQUEST / MISSION_REQUEST_INT
    received: set[int] = set()
    deadline = time.time() + UPLOAD_TIMEOUT * count
    while len(received) < count and time.time() < deadline:
        msg = master.recv_match(
            type=["MISSION_REQUEST", "MISSION_REQUEST_INT"],
            blocking=True, timeout=UPLOAD_TIMEOUT,
        )
        if msg is None:
            break
        seq  = msg.seq
        item = wp_loader.wp(seq)

        # DO/action commands must use MAV_FRAME_MISSION (2), not a GPS frame.
        # Using a GPS frame (3) on non-nav items causes PX4 to log
        # "IGN MISSION_ITEM: Invalid item" and may abort the upload.
        frame = item.frame
        if item.command not in _NAV_CMDS:
            frame = MAV_FRAME_MISSION

        if msg.get_type() == "MISSION_REQUEST_INT":
            # MISSION_ITEM_INT uses lat/lon as int32 × 1e7
            master.mav.mission_item_int_send(
                target_sys, target_comp,
                item.seq, frame, item.command,
                item.current, item.autocontinue,
                item.param1, item.param2, item.param3, item.param4,
                int(round(item.x * 1e7)), int(round(item.y * 1e7)), item.z,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )
        else:
            master.mav.mission_item_send(
                target_sys, target_comp,
                item.seq, frame, item.command,
                item.current, item.autocontinue,
                item.param1, item.param2, item.param3, item.param4,
                item.x, item.y, item.z,
            )
        received.add(seq)

    ack = master.recv_match(type="MISSION_ACK", blocking=True, timeout=UPLOAD_TIMEOUT)
    if ack is None or ack.type != mavutil.mavlink.MAV_MISSION_ACCEPTED:
        print(f"  [warn] Mission ACK: {ack}")
    else:
        print(f"  Mission upload ACK: ACCEPTED ({count} items)")

    # Count waypoints only (exclude CHANGE_SPEED, etc.)
    wp_cmds = {mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
               mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
               mavutil.mavlink.MAV_CMD_NAV_LAND}
    nav_count = sum(1 for i in range(count)
                    if wp_loader.wp(i).command in wp_cmds)
    return nav_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not WAYPOINTS_FILE.exists():
        print(f"ERROR: waypoints file not found:\n  {WAYPOINTS_FILE}", file=sys.stderr)
        sys.exit(1)

    # ── Load waypoints ────────────────────────────────────────────────────────
    wp_loader = mavwp.MAVWPLoader()
    wp_loader.load(str(WAYPOINTS_FILE))
    total_items = wp_loader.count()
    print(f"Loaded {total_items} mission items from {WAYPOINTS_FILE.name}")

    # ── Connect to PX4 SITL ───────────────────────────────────────────────────
    print(f"Connecting to PX4 SITL on {SITL_ADDR} …")
    master = mavutil.mavlink_connection(SITL_ADDR, source_system=255)
    wait_heartbeat(master, HEARTBEAT_TIMEOUT)

    # ── Configure airframe to match paper operating speed ─────────────────────
    configure_airframe(master)

    # ── Upload mission ────────────────────────────────────────────────────────
    print("Uploading mission …")
    nav_wp_count = upload_mission(master, wp_loader)
    print(f"  {nav_wp_count} navigation waypoints uploaded")

    # ── Wait for system ready ─────────────────────────────────────────────────
    wait_for_standby(master, READY_TIMEOUT)

    # ── Set AUTO / MISSION mode ───────────────────────────────────────────────
    print("Setting AUTO/MISSION mode …")
    set_mission_mode(master)
    print("  Mode set.")
    time.sleep(1)  # brief pause for mode transition

    # ── Arm ───────────────────────────────────────────────────────────────────
    print("Arming …")
    arm(master, ARM_TIMEOUT)
    print("  Armed.")
    t_start = time.time()
    t_last_wp = t_start  # updated each time a new WP is confirmed reached

    # ── Monitor mission progress ──────────────────────────────────────────────
    print("\nMonitoring mission (Ctrl-C to stop) …")
    print("-" * 56)
    reached: set[int] = set()   # unique waypoint indices reached
    last_reported = -1
    last_current = -1

    deadline = time.time() + MISSION_TIMEOUT
    while time.time() < deadline:
        msg = master.recv_match(
            type=["MISSION_ITEM_REACHED", "MISSION_CURRENT",
                  "STATUSTEXT", "HEARTBEAT"],
            blocking=True, timeout=5,
        )
        if msg is None:
            continue

        mtype = msg.get_type()

        if mtype == "MISSION_ITEM_REACHED":
            if msg.seq not in reached:
                reached.add(msg.seq)
                t_last_wp = time.time()
                elapsed = t_last_wp - t_start
                print(f"  Reached WP {msg.seq:3d}  [{elapsed:5.0f} s elapsed]"
                      f"  (unique: {len(reached)})")
                last_reported = msg.seq
                # Clean exit: all NAV waypoints confirmed reached
                if len(reached) >= nav_wp_count:
                    break

        elif mtype == "MISSION_CURRENT":
            if msg.seq != last_current:
                last_current = msg.seq

        elif mtype == "STATUSTEXT":
            text = getattr(msg, "text", "").strip()
            if text:
                print(f"  PX4: {text}")
            # PX4 emits "Mission finished" when done
            if "mission finished" in text.lower() or "mission complete" in text.lower():
                break

        elif mtype == "HEARTBEAT":
            # Check if vehicle is back in LOITER / disarmed (mission ended)
            if (msg.system_status == mavutil.mavlink.MAV_STATE_STANDBY
                    and len(reached) > 0):
                break

    # Use the last-WP timestamp so elapsed is not inflated by the timeout wait
    flight_end = t_last_wp if reached else time.time()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("-" * 56)
    print(f"Mission complete: {len(reached)} unique waypoints reached")
    print(f"Elapsed time    : {(flight_end - t_start) / 60:.2f} min  "
          f"(model prediction: 7.5 min)")
    if reached:
        sorted_reached = sorted(reached)
        print(f"WP range reached: {sorted_reached[0]} … {sorted_reached[-1]}")

    print("\nPX4 ULog written to:")
    log_root = Path(
        "/home/yliu/PX4-Autopilot/build/px4_sitl_default"
        "/rootfs/fs/microsd/log"
    )
    ulg_files = sorted(log_root.rglob("*.ulg")) if log_root.exists() else []
    if ulg_files:
        print(f"  {ulg_files[-1]}")
        print("\nRun next:")
        print("  python code/sitl_validation/03_plot_results.py")
    else:
        print(f"  (check {log_root})")


if __name__ == "__main__":
    main()
