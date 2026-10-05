"""
export.py
=========
Waypoint export to Mission Planner (.waypoints) and Google Earth (.kml) formats.

Mission Planner (.waypoints)
-----------------------------
Plain-text MAVLink WP 110 format.  Each line after the header is one waypoint:

    <seq> <current> <frame> <command> <param1..4> <x(lat)> <y(lon)> <z(alt)> <autocontinue>

Frame:  0 = MAV_FRAME_GLOBAL (absolute alt above MSL)
        3 = MAV_FRAME_GLOBAL_RELATIVE_ALT (alt relative to home, i.e. AGL)

We use frame=3 (AGL) throughout. All cells are MAV_CMD_NAV_WAYPOINT (16).
A MAV_CMD_DO_SET_CAM_TRIGG_DIST (206) is inserted as waypoint 1 to trigger
the camera at the required along-track spacing.

Google Earth KML (.kml)
-----------------------
Produces a KML with:
  - A Placemark/LineString showing the full flight path.
  - A Folder of Point Placemarks for every waypoint (toggle-able in GE).
  - A Placemark/Polygon showing the survey area outline.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

LonLat = Tuple[float, float]


# ---------------------------------------------------------------------------
# Mission Planner .waypoints (WP110)
# ---------------------------------------------------------------------------

MAV_CMD_NAV_WAYPOINT             = 16
MAV_CMD_DO_CHANGE_SPEED          = 178
MAV_CMD_DO_SET_CAM_TRIGG_DIST    = 206
MAV_FRAME_GLOBAL_RELATIVE_ALT    = 3


def write_waypoints(
    path: str | Path,
    lonlat_waypoints: List[LonLat],
    altitude_m: float,
    trigger_distance_m: float = 0.0,
    leg_speeds: List[tuple] | None = None,
    transition_speed_ms: float | None = None,
    trigger_waypoints: frozenset | None = None,
) -> None:
    """
    Write a Mission Planner .waypoints file.

    Parameters
    ----------
    path               : output file path (typically *.waypoints)
    lonlat_waypoints   : list of (lon, lat) flight waypoints in order
    altitude_m         : flight altitude AGL in metres
    trigger_distance_m : camera trigger distance in metres (0 = disabled)
    leg_speeds         : optional list of (start_idx, end_idx, speed_ms) tuples.
                         start_idx and end_idx are 0-based indices into
                         lonlat_waypoints.  A DO_CHANGE_SPEED command is inserted
                         before each segment's first waypoint.
    transition_speed_ms: deprecated; formerly used to detect transitions by speed
                         equality.  Ignored when trigger_waypoints is provided.
    trigger_waypoints  : frozenset of transition *destination* waypoint indices
                         (i.e. ``set(route.transition_indices)``).  When provided,
                         trigger-OFF is emitted before the departure waypoint of
                         each transition (index ti-1) and trigger-ON at the
                         transition destination (index ti), independent of speed.
                         Preferred over speed-equality detection.
    """
    lines: List[str] = ["QGC WPL 110"]

    # Waypoint 0: home / takeoff position (same as first survey waypoint)
    home_lon, home_lat = lonlat_waypoints[0]
    lines.append(
        f"0\t1\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t{MAV_CMD_NAV_WAYPOINT}\t"
        f"0\t0\t0\t0\t{home_lat:.8f}\t{home_lon:.8f}\t{altitude_m:.2f}\t1"
    )

    seq = 1

    # If no per-leg speed info, use legacy behaviour (single trigger command)
    if leg_speeds is None:
        if trigger_distance_m > 0:
            lines.append(
                f"{seq}\t0\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t"
                f"{MAV_CMD_DO_SET_CAM_TRIGG_DIST}\t"
                f"{trigger_distance_m:.2f}\t0\t1\t0\t0\t0\t0\t1"
            )
            seq += 1
        for lon, lat in lonlat_waypoints:
            lines.append(
                f"{seq}\t0\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t{MAV_CMD_NAV_WAYPOINT}\t"
                f"0\t0\t0\t0\t{lat:.8f}\t{lon:.8f}\t{altitude_m:.2f}\t1"
            )
            seq += 1
        Path(path).write_text("\n".join(lines) + "\n")
        return

    # Per-leg speed mode: build a segment map keyed by waypoint start index
    seg_by_start: dict[int, tuple] = {s: (s, e, sp) for (s, e, sp) in leg_speeds}

    # Pre-compute explicit trigger ON/OFF waypoints when transition_indices are
    # supplied.  This is independent of speed values and handles edge cases such
    # as single-cell final strips where no speed-run boundary fires.
    if trigger_waypoints is not None:
        _trig_offs: frozenset = frozenset(ti - 1 for ti in trigger_waypoints)
        _trig_ons:  frozenset = frozenset(trigger_waypoints)
    else:
        _trig_offs = frozenset()
        _trig_ons  = frozenset()

    current_trigger_on = False
    current_speed: float | None = None

    for wp_idx, (lon, lat) in enumerate(lonlat_waypoints):
        # ── Speed change (always driven by seg_by_start) ───────────────────
        seg_speed: float | None = None
        if wp_idx in seg_by_start:
            _, _, seg_speed = seg_by_start[wp_idx]
            if seg_speed != current_speed:
                lines.append(
                    f"{seq}\t0\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t"
                    f"{MAV_CMD_DO_CHANGE_SPEED}\t"
                    f"1\t{seg_speed:.2f}\t-1\t0\t0\t0\t0\t1"
                )
                seq += 1
                current_speed = seg_speed

        # ── Camera trigger on/off ──────────────────────────────────────────
        if trigger_distance_m > 0:
            if trigger_waypoints is not None:
                # Explicit mode: trigger-OFF at transition departure (ti-1),
                # trigger-ON at transition destination (ti) and at mission start.
                if wp_idx in _trig_offs:
                    if current_trigger_on:
                        lines.append(
                            f"{seq}\t0\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t"
                            f"{MAV_CMD_DO_SET_CAM_TRIGG_DIST}\t"
                            f"0\t0\t0\t0\t0\t0\t0\t1"
                        )
                        seq += 1
                        current_trigger_on = False
                elif wp_idx in _trig_ons or (wp_idx == 0 and not current_trigger_on):
                    if not current_trigger_on:
                        lines.append(
                            f"{seq}\t0\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t"
                            f"{MAV_CMD_DO_SET_CAM_TRIGG_DIST}\t"
                            f"{trigger_distance_m:.2f}\t0\t1\t0\t0\t0\t0\t1"
                        )
                        seq += 1
                        current_trigger_on = True
            else:
                # Legacy mode: speed-equality detection (used when
                # trigger_waypoints is not provided).
                if wp_idx in seg_by_start and seg_speed is not None:
                    is_transition = (
                        transition_speed_ms is not None
                        and abs(seg_speed - transition_speed_ms) < 0.01
                    )
                    want_trigger = not is_transition
                    if want_trigger and not current_trigger_on:
                        lines.append(
                            f"{seq}\t0\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t"
                            f"{MAV_CMD_DO_SET_CAM_TRIGG_DIST}\t"
                            f"{trigger_distance_m:.2f}\t0\t1\t0\t0\t0\t0\t1"
                        )
                        seq += 1
                        current_trigger_on = True
                    elif not want_trigger and current_trigger_on:
                        lines.append(
                            f"{seq}\t0\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t"
                            f"{MAV_CMD_DO_SET_CAM_TRIGG_DIST}\t"
                            f"0\t0\t0\t0\t0\t0\t0\t1"
                        )
                        seq += 1
                        current_trigger_on = False

        lines.append(
            f"{seq}\t0\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t{MAV_CMD_NAV_WAYPOINT}\t"
            f"0\t0\t0\t0\t{lat:.8f}\t{lon:.8f}\t{altitude_m:.2f}\t1"
        )
        seq += 1

    Path(path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Google Earth KML export
# ---------------------------------------------------------------------------

_KML_HEADER = """\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>{name}</name>
  <description>{description}</description>
  <Style id="pathStyle">
    <LineStyle><color>ff0000ff</color><width>2</width></LineStyle>
    <PolyStyle><fill>0</fill></PolyStyle>
  </Style>
  <Style id="areaStyle">
    <LineStyle><color>ff00ff00</color><width>2</width></LineStyle>
    <PolyStyle><color>3300ff00</color></PolyStyle>
  </Style>
  <Style id="wpStyle">
    <IconStyle>
      <color>ff0000ff</color>
      <scale>0.5</scale>
      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
    </IconStyle>
    <LabelStyle><scale>0</scale></LabelStyle>
  </Style>
"""

_KML_FOOTER = "</Document>\n</kml>\n"


def _coord_str(lonlat_list: List[LonLat], altitude_m: float = 0.0) -> str:
    return " ".join(f"{lon:.8f},{lat:.8f},{altitude_m:.2f}" for lon, lat in lonlat_list)


def write_kml(
    path: str | Path,
    lonlat_waypoints: List[LonLat],
    altitude_m: float,
    survey_polygon: List[LonLat] | None = None,
    name: str = "Coverage Flight Path",
    description: str = "",
) -> None:
    """
    Write a Google Earth KML file showing the flight path.

    Parameters
    ----------
    path              : output file path (*.kml)
    lonlat_waypoints  : ordered list of (lon, lat) flight waypoints
    altitude_m        : flight altitude AGL in metres (for path LineString)
    survey_polygon    : optional (lon, lat) polygon outline to overlay
    name              : document / placemark name
    description       : optional free-text description
    """
    body = _KML_HEADER.format(name=name, description=description)

    # Survey area polygon (optional)
    if survey_polygon:
        closed = list(survey_polygon)
        if closed[0] != closed[-1]:
            closed.append(closed[0])
        body += f"""  <Placemark>
    <name>Survey area</name>
    <styleUrl>#areaStyle</styleUrl>
    <Polygon>
      <outerBoundaryIs><LinearRing>
        <coordinates>{_coord_str(closed, 0.0)}</coordinates>
      </LinearRing></outerBoundaryIs>
    </Polygon>
  </Placemark>\n"""

    # Flight path LineString
    body += f"""  <Placemark>
    <name>{name}</name>
    <styleUrl>#pathStyle</styleUrl>
    <LineString>
      <altitudeMode>relativeToGround</altitudeMode>
      <tessellate>1</tessellate>
      <coordinates>{_coord_str(lonlat_waypoints, altitude_m)}</coordinates>
    </LineString>
  </Placemark>\n"""

    # Waypoint markers folder
    body += "  <Folder><name>Waypoints</name>\n"
    for i, (lon, lat) in enumerate(lonlat_waypoints):
        body += (
            f"    <Placemark><name>WP{i}</name><styleUrl>#wpStyle</styleUrl>"
            f"<Point><altitudeMode>relativeToGround</altitudeMode>"
            f"<coordinates>{lon:.8f},{lat:.8f},{altitude_m:.2f}</coordinates>"
            f"</Point></Placemark>\n"
        )
    body += "  </Folder>\n"

    body += _KML_FOOTER
    Path(path).write_text(body)
