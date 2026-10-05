"""
camera.py
=========
Camera footprint and cell-size calculation from altitude, FOV, and overlap settings.

Footprint model (flat-Earth, nadir-pointing camera):
    ground_width  = 2 * altitude * tan(FOV_h / 2)
    ground_height = 2 * altitude * tan(FOV_v / 2)

Effective cell dimensions after applying overlap:
    cell_w = ground_width  * (1 - overlap_side)
    cell_h = ground_height * (1 - overlap_front)

where overlap_side is the lateral (cross-track) overlap fraction and
overlap_front is the along-track overlap fraction.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Common sensor presets  (name -> (fov_h_deg, fov_v_deg))
# ---------------------------------------------------------------------------
SENSOR_PRESETS: dict[str, tuple[float, float]] = {
    # Sony A6000 + 16 mm lens (common mapping kit)
    "sony_a6000_16mm":   (73.7, 53.1),
    # DJI FC6310 (Phantom 4 Pro)
    "dji_fc6310":        (84.0, 70.0),
    # DJI Zenmuse X7 + 24 mm lens
    "zenmuse_x7_24mm":   (84.0, 61.0),
    # Generic wide–angle 90° sensor
    "generic_90deg":     (90.0, 67.5),
}


@dataclass
class CameraFootprint:
    """Ground footprint and derived cell dimensions."""
    fov_h_deg: float        # horizontal field-of-view in degrees
    fov_v_deg: float        # vertical field-of-view in degrees
    altitude_m: float       # flight altitude AGL in metres
    overlap_side: float     # lateral overlap fraction  [0, 1)
    overlap_front: float    # along-track overlap fraction  [0, 1)

    @property
    def ground_width_m(self) -> float:
        """Full swath width on the ground (metres)."""
        return 2.0 * self.altitude_m * math.tan(math.radians(self.fov_h_deg / 2.0))

    @property
    def ground_height_m(self) -> float:
        """Full swath height on the ground (metres)."""
        return 2.0 * self.altitude_m * math.tan(math.radians(self.fov_v_deg / 2.0))

    @property
    def cell_w(self) -> float:
        """Effective cross-track cell width after lateral overlap."""
        return self.ground_width_m * (1.0 - self.overlap_side)

    @property
    def cell_h(self) -> float:
        """Effective along-track cell height after forward overlap."""
        return self.ground_height_m * (1.0 - self.overlap_front)

    @property
    def trigger_distance_m(self) -> float:
        """Distance between camera triggers (= cell_h, along-track spacing)."""
        return self.cell_h

    def min_trigger_interval_s(self, camera_fps: float) -> float:
        """Minimum achievable trigger interval (seconds) given camera frame rate."""
        return 1.0 / camera_fps

    def max_groundspeed_for_camera(self, camera_fps: float) -> float:
        """Maximum groundspeed (m/s) that maintains overlap at the given fps.

        At this speed the drone advances exactly trigger_distance_m between
        consecutive frames, preserving the required forward overlap.
        """
        return self.trigger_distance_m * camera_fps

    def check_speed_feasibility(
        self, groundspeed_ms: float, camera_fps: float | None
    ) -> str | None:
        """Return a warning string if groundspeed may exceed camera capability.

        Returns None when no issue is detected or camera_fps is not provided.
        """
        if camera_fps is None:
            return None
        max_vg = self.max_groundspeed_for_camera(camera_fps)
        if groundspeed_ms > max_vg:
            interval = self.trigger_distance_m / groundspeed_ms
            return (
                f"WARNING: groundspeed {groundspeed_ms:.1f} m/s requires a trigger "
                f"interval of {interval:.2f} s, which is faster than the camera's "
                f"minimum ({1.0/camera_fps:.2f} s at {camera_fps:.0f} fps). "
                f"Consider reducing --max-groundspeed to {max_vg:.1f} m/s."
            )
        return None

    def summary(self) -> str:
        return (
            f"FOV {self.fov_h_deg:.1f}° × {self.fov_v_deg:.1f}°, "
            f"alt {self.altitude_m:.0f} m  →  "
            f"footprint {self.ground_width_m:.1f} m × {self.ground_height_m:.1f} m  →  "
            f"cell {self.cell_w:.1f} m × {self.cell_h:.1f} m  "
            f"(overlap {self.overlap_side*100:.0f}% side / {self.overlap_front*100:.0f}% front)"
        )


def camera_from_preset(
    preset: str,
    altitude_m: float,
    overlap_side: float = 0.70,
    overlap_front: float = 0.80,
) -> CameraFootprint:
    """Build a CameraFootprint from a named sensor preset."""
    if preset not in SENSOR_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. "
            f"Available: {list(SENSOR_PRESETS)}"
        )
    fov_h, fov_v = SENSOR_PRESETS[preset]
    return CameraFootprint(fov_h, fov_v, altitude_m, overlap_side, overlap_front)


def camera_from_fov(
    fov_h_deg: float,
    fov_v_deg: float,
    altitude_m: float,
    overlap_side: float = 0.70,
    overlap_front: float = 0.80,
) -> CameraFootprint:
    """Build a CameraFootprint from explicit FOV values."""
    return CameraFootprint(fov_h_deg, fov_v_deg, altitude_m, overlap_side, overlap_front)
