"""
coords.py
=========
Coordinate conversion between WGS-84 geographic coordinates and a local
UTM (metric) frame, using pyproj.

Workflow
--------
1. Call ``utm_proj_for_polygon(lon_lat_list)`` to obtain the best-fit UTM
   CRS for the polygon's centroid.
2. Use ``to_utm(lon_lat_list, crs)`` to project all vertices to metres.
3. Do all planning in the UTM frame (shapely Polygon with metre units).
4. Use ``from_utm(xy_list, crs)`` to convert waypoints back to (lon, lat).

KML parsing
-----------
``parse_kml_polygon(path)`` extracts the first Polygon coordinates from a
KML file saved by Google Earth and returns a list of (lon, lat) tuples.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple
from xml.etree import ElementTree as ET

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import Polygon

LonLat = Tuple[float, float]
XY = Tuple[float, float]

# ---------------------------------------------------------------------------
# KML parsing
# ---------------------------------------------------------------------------

# Google Earth KML namespaces
_KML_NS = {
    "kml": "http://www.opengis.net/kml/2.2",
    "kml21": "http://earth.google.com/kml/2.1",
}


def parse_kml_polygon(path: str | Path) -> List[LonLat]:
    """
    Parse the first Polygon in a KML file and return vertices as (lon, lat) list.

    Supports both KML 2.1 (Google Earth legacy) and KML 2.2 namespaces.
    The closing duplicate vertex (same as first) is removed if present.
    """
    tree = ET.parse(str(path))
    root = tree.getroot()

    # Detect namespace from root tag
    tag = root.tag
    if "}" in tag:
        ns_uri = tag[1:tag.index("}")]
        ns = {"kml": ns_uri}
    else:
        ns = {"kml": ""}

    def find_coords(element: ET.Element, ns_prefix: str) -> str | None:
        if ns_prefix:
            coord_tag = f"{{{ns_prefix}}}coordinates"
        else:
            coord_tag = "coordinates"
        for elem in element.iter(coord_tag):
            return elem.text
        return None

    ns_prefix = ns.get("kml", "")
    raw = find_coords(root, ns_prefix)
    if raw is None:
        # Try without namespace
        raw = find_coords(root, "")
    if raw is None:
        raise ValueError(f"No <coordinates> element found in KML file: {path}")

    vertices: List[LonLat] = []
    for token in raw.strip().split():
        parts = token.split(",")
        if len(parts) < 2:
            continue
        lon, lat = float(parts[0]), float(parts[1])
        vertices.append((lon, lat))

    if not vertices:
        raise ValueError(f"No valid coordinate triples found in KML file: {path}")

    # Remove closing duplicate
    if len(vertices) > 1 and vertices[0] == vertices[-1]:
        vertices = vertices[:-1]

    return vertices


# ---------------------------------------------------------------------------
# UTM projection
# ---------------------------------------------------------------------------

def utm_zone_for_lonlat(lon: float, lat: float) -> CRS:
    """Return the EPSG CRS for the UTM zone that contains (lon, lat)."""
    zone = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    if lat >= 0:
        epsg = 32600 + zone   # WGS-84 UTM north
    else:
        epsg = 32700 + zone   # WGS-84 UTM south
    return CRS.from_epsg(epsg)


def utm_proj_for_polygon(vertices: List[LonLat]) -> CRS:
    """Return a UTM CRS appropriate for the polygon centroid."""
    lons = [v[0] for v in vertices]
    lats = [v[1] for v in vertices]
    centre_lon = sum(lons) / len(lons)
    centre_lat = sum(lats) / len(lats)
    return utm_zone_for_lonlat(centre_lon, centre_lat)


def to_utm(vertices: List[LonLat], crs: CRS) -> List[XY]:
    """Project (lon, lat) vertices to (easting, northing) in the given UTM CRS."""
    wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(wgs84, crs, always_xy=True)
    return [transformer.transform(lon, lat) for lon, lat in vertices]


def from_utm(xy_list: List[XY] | np.ndarray, crs: CRS) -> List[LonLat]:
    """Back-project (easting, northing) to (lon, lat) in WGS-84."""
    wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs, wgs84, always_xy=True)
    result: List[LonLat] = []
    for xy in xy_list:
        lon, lat = transformer.transform(float(xy[0]), float(xy[1]))
        result.append((lon, lat))
    return result


def lonlat_to_utm_polygon(vertices: List[LonLat]) -> Tuple[Polygon, CRS]:
    """
    Convert a list of (lon, lat) vertices to a Shapely Polygon in UTM metres.

    Returns
    -------
    polygon_utm : Shapely Polygon in metres
    crs         : UTM CRS used (needed for back-projection)
    """
    crs = utm_proj_for_polygon(vertices)
    xy = to_utm(vertices, crs)
    return Polygon(xy), crs
