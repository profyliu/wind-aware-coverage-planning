"""
decomposition.py
================
Selective uniform-grid decomposition for polygon coverage.

Implements the any-intersection boundary-retention rule:
    retain cell Q  iff  Q ∩ P ≠ ∅

Key public functions
--------------------
decompose(polygon, cell_w, cell_h, angle_deg)
    -> DecompositionResult (retained cells, adjacency graph, metadata)

Definition 1 (Boundary-Retention Rule, as stated in the paper)
--------------------------------------------------------------
A grid cell Q is retained if and only if Q ∩ P ≠ ∅.
For a retained boundary cell, the UAV executes the full rectangular sweep
over the cell footprint Q (not just over Q ∩ P), so overshoot outside P is
accepted in the operational model.

Lemma 1 (Coverage under the any-intersection rule)
---------------------------------------------------
P ⊆ ∪_{c ∈ R} c  under the any-intersection rule.
Proof sketch: every x ∈ P lies in some tile Q_ab;  x ∈ Q_ab ∩ P implies
Q_ab ∩ P ≠ ∅, so Q_ab ∈ R; therefore x ∈ ∪_{c ∈ R} c.  □
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Point, Polygon, box


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    """A retained grid cell.

    Attributes
    ----------
    row, col   : integer grid indices in the rotated frame
    footprint  : shapely Polygon – the full rectangular cell in original coords
    is_boundary: True if cell is only partially inside P
    center     : representative point (cell centroid) in original coords
    intersection_area: area of footprint ∩ P
    """
    row: int
    col: int
    footprint: Polygon
    is_boundary: bool
    center: np.ndarray
    intersection_area: float
    strip_id: Optional[int] = None   # assigned by routing


@dataclass
class DecompositionResult:
    """Output of decompose()."""
    polygon: Polygon
    angle_deg: float
    cell_w: float
    cell_h: float
    cells: List[Cell] = field(default_factory=list)
    adjacency: nx.Graph = field(default_factory=nx.Graph)
    # Mapping (row, col) -> cell index in self.cells
    index_map: Dict[Tuple[int, int], int] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def coverage_ratio(self) -> float:
        """Fraction of P covered by the union of retained cells."""
        if self.polygon.area == 0:
            return 1.0
        from shapely.ops import unary_union
        union = unary_union([c.footprint for c in self.cells])
        covered = union.intersection(self.polygon).area
        return covered / self.polygon.area

    @property
    def n_boundary_cells(self) -> int:
        return sum(1 for c in self.cells if c.is_boundary)

    @property
    def n_interior_cells(self) -> int:
        return sum(1 for c in self.cells if not c.is_boundary)


# ---------------------------------------------------------------------------
# Core decomposition
# ---------------------------------------------------------------------------

def decompose(polygon: Polygon,
              cell_w: float,
              cell_h: float,
              angle_deg: float = 0.0) -> DecompositionResult:
    """
    Decompose *polygon* into a selective uniform grid with any-intersection rule.

    Parameters
    ----------
    polygon   : target region (simple connected polygon, no holes assumed)
    cell_w    : cell width in metres (in the rotated / wind-aligned frame)
    cell_h    : cell height in metres
    angle_deg : grid orientation angle in degrees (wind direction angle θ);
                the grid axes are aligned with this direction.

    Returns
    -------
    DecompositionResult with all retained cells and the side-adjacency graph.
    """
    if not polygon.is_valid:
        from shapely.validation import make_valid
        polygon = make_valid(polygon)

    result = DecompositionResult(
        polygon=polygon,
        angle_deg=angle_deg,
        cell_w=cell_w,
        cell_h=cell_h,
    )

    # ------------------------------------------------------------------
    # Step 1: rotate polygon to wind-aligned frame (angle = -theta)
    # ------------------------------------------------------------------
    cx, cy = polygon.centroid.x, polygon.centroid.y
    poly_rot = rotate(polygon, -angle_deg, origin=(cx, cy), use_radians=False)

    # ------------------------------------------------------------------
    # Step 2: build uniform grid over the bounding box of the rotated polygon
    # ------------------------------------------------------------------
    minx, miny, maxx, maxy = poly_rot.bounds
    # Align grid origin to a multiple of cell dimensions so indices are stable
    origin_x = math.floor(minx / cell_w) * cell_w
    origin_y = math.floor(miny / cell_h) * cell_h

    col_min = int(math.floor((minx - origin_x) / cell_w)) - 1
    col_max = int(math.ceil((maxx - origin_x) / cell_w)) + 1
    row_min = int(math.floor((miny - origin_y) / cell_h)) - 1
    row_max = int(math.ceil((maxy - origin_y) / cell_h)) + 1

    # ------------------------------------------------------------------
    # Step 3 & 4: apply any-intersection retention rule
    # ------------------------------------------------------------------
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            # Cell in rotated frame
            x0 = origin_x + col * cell_w
            y0 = origin_y + row * cell_h
            cell_rot = box(x0, y0, x0 + cell_w, y0 + cell_h)

            # Any-intersection rule: retain iff intersection is non-empty
            inter = cell_rot.intersection(poly_rot)
            if inter.is_empty:
                continue

            # Rotate cell footprint back to original frame
            cell_orig = rotate(cell_rot, angle_deg, origin=(cx, cy), use_radians=False)

            inter_area = inter.area
            total_area = cell_rot.area
            is_boundary = inter_area < total_area - 1e-9

            # Representative point: centroid of the cell footprint (original frame)
            centroid = np.array([cell_orig.centroid.x, cell_orig.centroid.y])

            cell_obj = Cell(
                row=row,
                col=col,
                footprint=cell_orig,
                is_boundary=is_boundary,
                center=centroid,
                intersection_area=inter_area,
            )
            idx = len(result.cells)
            result.cells.append(cell_obj)
            result.index_map[(row, col)] = idx

    # ------------------------------------------------------------------
    # Step 5: build side-adjacency graph (shared full-edge adjacency)
    # ------------------------------------------------------------------
    G = nx.Graph()
    G.add_nodes_from(range(result.n_cells))

    for (row, col), idx in result.index_map.items():
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nb = (row + dr, col + dc)
            if nb in result.index_map:
                nb_idx = result.index_map[nb]
                if not G.has_edge(idx, nb_idx):
                    G.add_edge(idx, nb_idx)

    result.adjacency = G
    return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def verify_coverage(result: DecompositionResult) -> bool:
    """
    Verify Lemma 1: P ⊆ union of retained cells.
    Returns True if coverage ratio ≥ 1 - 1e-6.
    """
    ratio = result.coverage_ratio
    return ratio >= 1.0 - 1e-6


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from instance_gen import generate_polygon_family

    for fam in ["convex", "mild_concave", "high_concave", "corridor"]:
        poly = generate_polygon_family(fam, seed=0)
        res = decompose(poly, cell_w=20.0, cell_h=20.0, angle_deg=45.0)
        ok = verify_coverage(res)
        print(f"{fam:15s}: n_cells={res.n_cells:4d}  "
              f"boundary={res.n_boundary_cells:3d}  "
              f"interior={res.n_interior_cells:3d}  "
              f"coverage={res.coverage_ratio:.6f}  "
              f"lemma1={'OK' if ok else 'FAIL'}")
