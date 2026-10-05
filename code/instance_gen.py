"""
instance_gen.py
===============
Polygon instance generation for the CPP benchmark study.

Generates four polygon families used in the experiments:
  - convex       : convex hulls of points on a perturbed circle
  - mild_concave : convex hull with inward notches on 10-20% of edges
  - high_concave : deeper indentations (3-8 notches)
  - corridor     : chambers connected by narrow passages

All coordinates are in metres.  Each polygon is returned as a
shapely.geometry.Polygon object with the bounding-box diagonal
normalized to a requested scale (default 500 m).

Usage
-----
>>> from instance_gen import generate_polygon_family
>>> polys = generate_polygon_family("convex", n_vertices=10, seed=0)
"""

import math
import random
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _normalize_scale(poly: Polygon, target_diagonal: float = 500.0) -> Polygon:
    """Scale polygon so that bounding-box diagonal equals target_diagonal."""
    minx, miny, maxx, maxy = poly.bounds
    diag = math.hypot(maxx - minx, maxy - miny)
    if diag == 0:
        return poly
    scale = target_diagonal / diag
    coords = np.array(poly.exterior.coords) * scale
    return Polygon(coords)


def _perturb_circle_points(n: int, rng: random.Random, roughness: float = 0.05) -> np.ndarray:
    """Return n points on a circle perturbed by up to roughness (as fraction of radius)."""
    angles = sorted([rng.uniform(0, 2 * math.pi) for _ in range(n)])
    pts = []
    for a in angles:
        r = 1.0 + rng.uniform(-roughness, roughness)
        pts.append([r * math.cos(a), r * math.sin(a)])
    return np.array(pts)


# ---------------------------------------------------------------------------
# Family generators
# ---------------------------------------------------------------------------

def generate_convex(n_vertices: int = 10, seed: int = 0,
                    roughness: float = 0.03,
                    target_diagonal: float = 500.0) -> Polygon:
    """Convex polygon with n_vertices on a perturbed circle."""
    rng = random.Random(seed)
    pts = _perturb_circle_points(n_vertices, rng, roughness)
    poly = Polygon(pts).convex_hull
    return _normalize_scale(poly, target_diagonal)


def generate_mild_concave(n_vertices: int = 12, seed: int = 0,
                           notch_fraction: float = 0.15,
                           notch_depth: float = 0.15,
                           target_diagonal: float = 500.0) -> Polygon:
    """
    Convex polygon with inward notches inserted on a fraction of edges.

    notch_fraction : fraction of edges that get a notch (0.1–0.2 recommended)
    notch_depth    : notch depth as fraction of local edge length
    """
    rng = random.Random(seed)
    pts = _perturb_circle_points(n_vertices, rng, roughness=0.02)
    hull_coords = list(Polygon(pts).convex_hull.exterior.coords[:-1])

    new_coords = []
    n = len(hull_coords)
    for i in range(n):
        p1 = np.array(hull_coords[i])
        p2 = np.array(hull_coords[(i + 1) % n])
        new_coords.append(p1.tolist())
        if rng.random() < notch_fraction:
            mid = (p1 + p2) / 2
            edge_vec = p2 - p1
            # Inward normal (leftward for CCW polygon)
            inward = np.array([-edge_vec[1], edge_vec[0]])
            inward /= np.linalg.norm(inward)
            depth = notch_depth * np.linalg.norm(edge_vec)
            notch_pt = mid - inward * depth  # push inward (subtract outward)
            new_coords.append(notch_pt.tolist())

    poly = Polygon(new_coords)
    if not poly.is_valid:
        poly = make_valid(poly)
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
    return _normalize_scale(poly, target_diagonal)


def generate_high_concave(n_notches: int = 5, seed: int = 0,
                           notch_depth: float = 0.30,
                           target_diagonal: float = 500.0) -> Polygon:
    """
    Roughly circular polygon with n_notches deep indentations.
    """
    rng = random.Random(seed)
    base_n = max(16, n_notches * 4)
    pts = _perturb_circle_points(base_n, rng, roughness=0.02)
    hull_coords = list(Polygon(pts).convex_hull.exterior.coords[:-1])
    n = len(hull_coords)

    # Pick n_notches evenly spaced edges to indent
    notch_edges = sorted(rng.sample(range(n), min(n_notches, n)))
    notch_set = set(notch_edges)

    new_coords = []
    for i in range(n):
        p1 = np.array(hull_coords[i])
        p2 = np.array(hull_coords[(i + 1) % n])
        new_coords.append(p1.tolist())
        if i in notch_set:
            mid = (p1 + p2) / 2
            edge_vec = p2 - p1
            inward = np.array([-edge_vec[1], edge_vec[0]])
            norm = np.linalg.norm(inward)
            if norm > 1e-10:
                inward /= norm
                depth = notch_depth * np.linalg.norm(edge_vec)
                notch_pt = mid - inward * depth
                new_coords.append(notch_pt.tolist())

    poly = Polygon(new_coords)
    if not poly.is_valid:
        poly = make_valid(poly)
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
    return _normalize_scale(poly, target_diagonal)


def generate_corridor(n_chambers: int = 2, corridor_width: float = 40.0,
                      seed: int = 0,
                      target_diagonal: float = 500.0) -> Polygon:
    """
    Two rectangular chambers connected by a narrow horizontal corridor.

    corridor_width : width of the connecting passage in the normalized frame (metres)
    """
    rng = random.Random(seed)
    chamber_w = 150.0 + rng.uniform(-20, 20)
    chamber_h = 200.0 + rng.uniform(-20, 20)
    gap = 80.0 + rng.uniform(-10, 10)  # horizontal separation between chambers
    corr_h = corridor_width

    # Left chamber
    lx0, ly0 = 0.0, 0.0
    lx1, ly1 = chamber_w, chamber_h

    # Right chamber
    rx0 = lx1 + gap
    ry0 = 0.0
    rx1 = rx0 + chamber_w
    ry1 = chamber_h

    # Corridor (connecting mid-heights)
    cy0 = (chamber_h - corr_h) / 2
    cy1 = cy0 + corr_h

    # Build union polygon as explicit coordinate list (clockwise outer boundary)
    coords = [
        (lx0, ly0), (lx0, ly1), (lx1, ly1), (lx1, cy1),
        (rx0, cy1), (rx0, ry1), (rx1, ry1), (rx1, ry0),
        (rx0, ry0), (rx0, cy0), (lx1, cy0), (lx1, ly0),
    ]
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = make_valid(poly)
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
    return _normalize_scale(poly, target_diagonal)


# ---------------------------------------------------------------------------
# Family batch generator
# ---------------------------------------------------------------------------

FAMILY_PARAMS = {
    "convex": {
        "n_vertices_list": [6, 10, 14, 18],
        "roughness": 0.03,
    },
    "mild_concave": {
        "n_vertices_list": [10, 14, 18, 24],
        "notch_fraction": 0.15,
        "notch_depth": 0.15,
    },
    "high_concave": {
        "n_notches_list": [3, 5, 7, 8],
        "notch_depth": 0.30,
    },
    "corridor": {
        "corridor_widths": [30.0, 40.0, 60.0, 80.0],
    },
}


def generate_polygon_family(family: str, seed: int = 0,
                              target_diagonal: float = 500.0, **kwargs) -> Polygon:
    """
    Generate a single polygon from the named family.

    Parameters
    ----------
    family : one of "convex", "mild_concave", "high_concave", "corridor"
    seed   : random seed for reproducibility
    **kwargs: passed to the underlying generator (e.g. n_vertices, notch_depth)
    """
    if family == "convex":
        n = kwargs.get("n_vertices", 10)
        roughness = kwargs.get("roughness", 0.03)
        return generate_convex(n, seed, roughness, target_diagonal)
    elif family == "mild_concave":
        n = kwargs.get("n_vertices", 14)
        nf = kwargs.get("notch_fraction", 0.15)
        nd = kwargs.get("notch_depth", 0.15)
        return generate_mild_concave(n, seed, nf, nd, target_diagonal)
    elif family == "high_concave":
        nn = kwargs.get("n_notches", 5)
        nd = kwargs.get("notch_depth", 0.30)
        return generate_high_concave(nn, seed, nd, target_diagonal)
    elif family == "corridor":
        cw = kwargs.get("corridor_width", 40.0)
        return generate_corridor(seed=seed, corridor_width=cw,
                                  target_diagonal=target_diagonal)
    else:
        raise ValueError(f"Unknown polygon family: {family!r}")


def load_scenario_polygon(scenario_path: str) -> Polygon:
    """Load polygon vertices from a scenario JSON file."""
    import json
    with open(scenario_path) as f:
        data = json.load(f)
    verts = data["polygon_vertices"]
    return Polygon(verts)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for fam in ["convex", "mild_concave", "high_concave", "corridor"]:
        p = generate_polygon_family(fam, seed=42)
        print(f"{fam:15s}: area={p.area:.0f} m², "
              f"bbox={p.bounds[2]-p.bounds[0]:.0f}×"
              f"{p.bounds[3]-p.bounds[1]:.0f} m, "
              f"valid={p.is_valid}")
