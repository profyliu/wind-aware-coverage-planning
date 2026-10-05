"""
routing.py
==========
Strip-based wind-aware coverage routing.

Implements Algorithm 1 Steps 5–12 from the Methods section:
  - Partition retained cells into strips (maximal contiguous row-runs)
  - Choose traversal direction per strip using the active cost function
    (travel time or propulsion energy)
  - Build greedy and boustrophedon initial strip orders
  - Improve the order with Or-opt-1 and 2-opt local search
  - Concatenate coverage primitives and transition segments

Ground-speed model (committed, as stated in the Problem section)
---------------------------------------------------------------
The UAV crabs into the crosswind to maintain the desired ground track.
Decomposing the wind into along-track (w_∥ = v_w cosθ) and cross-track
(w_⊥ = v_w |sinθ|) components and imposing constant airspeed v_a:

    vg_ij = v_w·cos(θ_ij) + sqrt(v_a² − v_w²·sin²(θ_ij))

where θ_ij is the angle between the desired ground track and the wind.
Feasibility condition: v_a > v_w  → discriminant v_a²−v_w²sin²θ ≥ v_a²−v_w² > 0.

Numeric example: v_a=12 m/s, v_w=3 m/s
    - tailwind  (θ=0°)  → vg = 3 + sqrt(144)      = 15.00 m/s
    - crosswind (θ=90°) → vg = 0 + sqrt(135)      ≈ 11.62 m/s
    - headwind  (θ=180°)→ vg = -3 + sqrt(144)     =  9.00 m/s
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from decomposition import Cell, DecompositionResult, decompose as _decompose_for_sweep


# ---------------------------------------------------------------------------
# Wind-aware travel-time model
# ---------------------------------------------------------------------------

def travel_time(p1: np.ndarray, p2: np.ndarray,
                wind_vec: np.ndarray, airspeed: float) -> float:
    """
    Compute travel time from p1 to p2 under steady wind.

    Parameters
    ----------
    p1, p2    : 2-D position vectors (metres)
    wind_vec  : horizontal wind vector [vwx, vwy] (m/s)
    airspeed  : UAV airspeed relative to air (m/s); must satisfy airspeed > |wind_vec|

    Returns
    -------
    Travel time in seconds.  Returns inf if denominator ≤ 0 (headwind exceeds airspeed).
    """
    disp = p2 - p1
    dist = np.linalg.norm(disp)
    if dist < 1e-10:
        return 0.0
    u_hat = disp / dist
    # Along-track wind component (positive = tailwind)
    w_parallel = float(np.dot(wind_vec, u_hat))
    # Cross-track wind component (squared; always non-negative)
    w_perp_sq = float(np.dot(wind_vec, wind_vec)) - w_parallel ** 2
    # Discriminant is v_a² - v_w²sin²θ = airspeed² - w_perp²
    discriminant = airspeed ** 2 - w_perp_sq
    if discriminant <= 0:
        return math.inf  # infeasible: crosswind exceeds airspeed (should not occur if v_a > v_w)
    vg = w_parallel + math.sqrt(discriminant)
    if vg <= 0:
        return math.inf
    return dist / vg


# ---------------------------------------------------------------------------
# Rotary-wing propulsion power model
# ---------------------------------------------------------------------------

@dataclass
class PowerModel:
    """
    Rotary-wing propulsion power model (Zeng, Xu, Zhang 2019).

    The shaft power at horizontal airspeed v_a (m/s) is

        P(v_a) = P0 * (1 + 3*v_a² / U_tip²)
               + Pi * (sqrt(1 + v_a⁴ / (4*v0⁴)) − v_a² / (2*v0²))^{1/2}
               + 0.5 * rho * C_D * A * v_a³

    where
      P0    [W]        blade-profile power in hover (blade-element/momentum theory)
      Pi    [W]        induced power in hover
      U_tip [m/s]      blade-tip speed
      v0    [m/s]      mean rotor-disc induced velocity in hover
      rho   [kg/m³]    air density (1.225 at sea level, ISA)
      C_D   [–]        equivalent fuselage drag coefficient
      A     [m²]       reference (frontal) area of the fuselage

    The model has a U-shaped power-vs-speed profile: induced power decreases
    with speed while parasitic drag power rises as v³, yielding a well-defined
    minimum-energy-per-distance (maximum-range) airspeed.

    Calibrated to a DJI Phantom 4 class mapping UAV (~1.5 kg MTOW):
        P0 = 79.86 W,  Pi = 88.63 W
        U_tip = 120 m/s,  v0 = 4.03 m/s
        rho = 1.225 kg/m³,  C_D = 0.6,  A = 0.0151 m²
      → hover power ≈ 168 W;  cruise power at 12 m/s ≈ 121 W

    Reference: Y. Zeng, J. Xu, R. Zhang, "Energy Minimization for Wireless
    Communication with Rotary-Wing UAV," IEEE Trans. Wireless Commun., 2019.
    """

    P0: float           # blade-profile hover power      [W]
    Pi: float           # induced hover power             [W]
    U_tip: float        # blade-tip speed                 [m/s]
    v0: float           # induced velocity in hover       [m/s]
    rho: float = 1.225  # air density                     [kg/m³]
    C_D: float = 0.6    # fuselage drag coefficient       [–]
    A: float = 0.0151   # reference frontal area          [m²]
    v_a_min: float = 8.0     # minimum operational airspeed (m/s)
    v_a_max: float = 25.0    # maximum operational airspeed (m/s)

    def power(self, v_a: float) -> float:
        """Shaft power in Watts at airspeed v_a (m/s)."""
        va2 = v_a * v_a
        va4 = va2 * va2
        v02 = self.v0 * self.v0
        v04 = v02 * v02
        sq_term = math.sqrt(max(1.0 + va4 / (4.0 * v04), 1.0))
        inner = max(sq_term - va2 / (2.0 * v02), 0.0)   # numerical guard at hover
        profile   = self.P0 * (1.0 + 3.0 * va2 / (self.U_tip ** 2))
        induced   = self.Pi * math.sqrt(inner)
        parasitic = 0.5 * self.rho * self.C_D * self.A * va2 * v_a
        return profile + induced + parasitic

    @property
    def v_opt_calm(self) -> float:
        """Energy-optimal airspeed in calm air (m/s): minimises P(v_a)/v_a.
        Computed numerically by golden-section search on [v_a_min, v_a_max]."""
        return _golden_section_min(
            lambda va: self.power(va) / va if va > 0 else math.inf,
            self.v_a_min, self.v_a_max,
        )


# Default model — DJI Phantom 4 class mapping multirotor (~1.5 kg MTOW)
# Parameters from Zeng et al. (2019) blade-element/momentum theory
DEFAULT_POWER_MODEL = PowerModel(
    P0=79.86, Pi=88.63, U_tip=120.0, v0=4.03,
    rho=1.225, C_D=0.6, A=0.0151,
    v_a_min=8.0, v_a_max=25.0,
)


def _golden_section_min(f, a: float, b: float,
                        tol: float = 1e-7, max_iter: int = 100) -> float:
    """Golden-section search for the minimum of a unimodal function on [a, b]."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0   # golden ratio ≈ 0.618
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
        if b - a < tol:
            break
    return (a + b) / 2.0


def optimal_airspeed(theta_rad: float, wind_speed: float,
                     power_model: PowerModel) -> float:
    """
    Find the energy-optimal airspeed v_a*(θ) by golden-section minimisation of
    the energy-per-unit-distance e(v_a, θ) = P(v_a) / vg(v_a, θ).

    For a leg flown at heading angle θ relative to the wind, the energy per unit
    distance is e(v_a, θ) = P(v_a) / vg(v_a, θ).  The minimiser satisfies

        ∂e/∂v_a = 0   ⟺   dP/dv_a · vg = P · dvg/dv_a

    For the rotary-wing power model this has no closed-form solution, so
    golden-section search is applied on [v_a_min, v_a_max]
    (Proposition~2 of the paper).

    Special case: in calm air (v_w = 0) the optimal speed equals
    v_opt_calm = power_model.v_opt_calm.

    Parameters
    ----------
    theta_rad   : angle between leg heading and wind vector (radians)
    wind_speed  : wind speed |v_w| (m/s)
    power_model : PowerModel instance

    Returns
    -------
    v_a_opt : float — energy-optimal airspeed in m/s
    """
    w_along = wind_speed * math.cos(theta_rad)
    w_perp_sq = (wind_speed * math.sin(theta_rad)) ** 2

    def energy_per_m(va: float) -> float:
        disc = va * va - w_perp_sq
        if disc <= 0.0:
            return math.inf
        vg = w_along + math.sqrt(disc)
        if vg <= 0.0:
            return math.inf
        return power_model.power(va) / vg

    # Lower bound must exceed |w_perp| to keep the leg feasible
    v_lo = max(power_model.v_a_min, math.sqrt(w_perp_sq) + 1e-4)
    v_hi = power_model.v_a_max
    if v_lo >= v_hi:
        return v_lo

    return _golden_section_min(energy_per_m, v_lo, v_hi)


def energy_cost(p1: np.ndarray, p2: np.ndarray,
                wind_vec: np.ndarray, power_model: PowerModel) -> float:
    """
    Minimum energy (Joules) to fly from p1 to p2 at the per-leg optimal airspeed.

    The optimal airspeed v_a*(θ) is determined by golden-section search (see optimal_airspeed).
    The result equals P(v_a*(θ)) · d / vg(v_a*(θ), θ), where d = ‖p2 − p1‖.
    """
    disp = p2 - p1
    dist = float(np.linalg.norm(disp))
    if dist < 1e-10:
        return 0.0

    u_hat = disp / dist
    w_along = float(np.dot(wind_vec, u_hat))
    w_perp_sq = float(np.dot(wind_vec, wind_vec)) - w_along ** 2
    wind_speed = float(np.linalg.norm(wind_vec))
    theta_rad = math.atan2(math.sqrt(max(w_perp_sq, 0.0)), w_along)

    va_opt = optimal_airspeed(theta_rad, wind_speed, power_model)
    disc = va_opt ** 2 - w_perp_sq
    if disc <= 0:
        return math.inf
    vg = w_along + math.sqrt(disc)
    if vg <= 0:
        return math.inf

    t = dist / vg
    return power_model.power(va_opt) * t


def _leg_cost(p1: np.ndarray, p2: np.ndarray,
              wind_vec: np.ndarray, airspeed: float,
              power_model: Optional["PowerModel"],
              objective: str = "time") -> float:
    """
    Unified leg-cost dispatcher used by all routing decisions.

    The *objective* parameter controls what is *optimised*:
      - ``'time'``  → returns travel time  (seconds)
      - ``'energy'`` → returns minimum energy (Joules) using *power_model*

    *power_model* is also used separately for energy *reporting* in
    ``build_route``; passing a model with ``objective='time'`` therefore
    enables full energy-metric reporting while keeping time as the routing
    criterion.  Routing decisions always call this function, so switching
    *objective* propagates automatically through all decision points.
    """
    if objective == "energy":
        assert power_model is not None, "objective='energy' requires a PowerModel"
        return energy_cost(p1, p2, wind_vec, power_model)
    return travel_time(p1, p2, wind_vec, airspeed)


# ---------------------------------------------------------------------------
# Strip data structure
# ---------------------------------------------------------------------------

@dataclass
class Strip:
    """A maximal contiguous row-run of retained cells."""
    strip_id: int
    row: int
    cells: List[Cell]          # sorted by increasing column index
    direction: int = 1         # +1 = left-to-right,  -1 = right-to-left
    entry_point: Optional[np.ndarray] = None
    exit_point: Optional[np.ndarray] = None

    @property
    def col_start(self) -> int:
        return self.cells[0].col

    @property
    def col_end(self) -> int:
        return self.cells[-1].col

    def compute_entry_exit(self) -> None:
        """Set entry/exit points based on chosen traversal direction."""
        if self.direction == 1:   # left-to-right
            self.entry_point = self.cells[0].center.copy()
            self.exit_point = self.cells[-1].center.copy()
        else:                     # right-to-left
            self.entry_point = self.cells[-1].center.copy()
            self.exit_point = self.cells[0].center.copy()


# ---------------------------------------------------------------------------
# Strip ordering helpers  (boustrophedon init + 2-opt + Or-opt-1)
# ---------------------------------------------------------------------------

def _copy_strip(s: Strip) -> Strip:
    """Shallow copy of Strip with independent numpy arrays for entry/exit."""
    ns = copy.copy(s)
    if ns.entry_point is not None:
        ns.entry_point = ns.entry_point.copy()
    if ns.exit_point is not None:
        ns.exit_point = ns.exit_point.copy()
    # cells list is shared (cells are never mutated by routing)
    return ns


def _transition_total(strips: List[Strip],
                      wind_vec: np.ndarray, airspeed: float,
                      power_model: Optional["PowerModel"] = None,
                      objective: str = "time") -> float:
    """Total transition cost (time or energy) for a strip sequence."""
    if len(strips) <= 1:
        return 0.0
    return sum(
        _leg_cost(strips[k].exit_point, strips[k + 1].entry_point,
                  wind_vec, airspeed, power_model, objective)
        for k in range(len(strips) - 1)
    )


def _boustrophedon_init(strips: List[Strip],
                        wind_vec: np.ndarray, airspeed: float,
                        power_model: Optional["PowerModel"] = None,
                        objective: str = "time") -> List[Strip]:
    """
    Row-by-row serpentine (boustrophedon) strip ordering.
    Even-indexed rows go left-to-right; odd-indexed rows go right-to-left.
    Each strip's traversal direction is chosen greedily to minimise the
    transition from the previous exit point using the active cost function.
    """
    by_row: Dict[int, List[Strip]] = {}
    for s in strips:
        by_row.setdefault(s.row, []).append(s)

    ordered: List[Strip] = []
    for i, row in enumerate(sorted(by_row.keys())):
        row_strips = sorted(by_row[row], key=lambda s: s.col_start)
        if i % 2 == 1:
            row_strips = list(reversed(row_strips))
        for s in row_strips:
            if ordered:
                prev_exit = ordered[-1].exit_point
                c_fwd = _leg_cost(prev_exit, s.cells[0].center,
                                  wind_vec, airspeed, power_model, objective)
                c_rev = _leg_cost(prev_exit, s.cells[-1].center,
                                  wind_vec, airspeed, power_model, objective)
                s.direction = 1 if c_fwd <= c_rev else -1
                s.compute_entry_exit()
            ordered.append(s)
    return ordered


def _two_opt(strips: List[Strip],
             wind_vec: np.ndarray, airspeed: float,
             max_passes: int = 200,
             power_model: Optional["PowerModel"] = None,
             objective: str = "time") -> List[Strip]:
    """
    2-opt improvement for an open strip-ordering path.

    For every pair (i, j) with i < j, evaluates reversing the sub-sequence
    strips[i : j+1] — each strip in the reversed segment has its traversal
    direction flipped so that entry/exit are consistent after reordering.

    Uses precomputed cumulative forward/reverse transition sums so that each
    (i, j) pair is evaluated in O(1), keeping the full pass at O(r²).
    """
    n = len(strips)
    if n <= 2:
        return strips

    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1

        # fwd[k] = cost(strips[k].exit → strips[k+1].entry)  (current order)
        # rev[k] = cost(strips[k+1].entry → strips[k].exit)  (reversed segment)
        fwd = [
            _leg_cost(strips[k].exit_point, strips[k + 1].entry_point,
                      wind_vec, airspeed, power_model, objective)
            for k in range(n - 1)
        ]
        rev = [
            _leg_cost(strips[k + 1].entry_point, strips[k].exit_point,
                      wind_vec, airspeed, power_model, objective)
            for k in range(n - 1)
        ]
        # Prefix sums for O(1) internal-transition queries
        cum_fwd = [0.0] * n
        cum_rev = [0.0] * n
        for k in range(n - 1):
            cum_fwd[k + 1] = cum_fwd[k] + fwd[k]
            cum_rev[k + 1] = cum_rev[k] + rev[k]

        for i in range(n - 1):
            for j in range(i + 1, n):
                # Boundary transitions removed by the reversal
                old_b = (
                    (fwd[i - 1] if i > 0 else 0.0) +
                    (fwd[j]     if j < n - 1 else 0.0)
                )
                # Boundary transitions added after reversal:
                #   strips[j] is now first in the reversed block (direction
                #   flipped), so its new "entry" = its old exit_point.
                #   strips[i] is now last (flipped), so its new "exit"
                #   = its old entry_point.
                new_b = (
                    (_leg_cost(strips[i - 1].exit_point, strips[j].exit_point,
                               wind_vec, airspeed, power_model, objective) if i > 0 else 0.0) +
                    (_leg_cost(strips[i].entry_point, strips[j + 1].entry_point,
                               wind_vec, airspeed, power_model, objective) if j < n - 1 else 0.0)
                )
                # Internal transitions (O(1) via prefix sums)
                old_int = cum_fwd[j] - cum_fwd[i]
                new_int = cum_rev[j] - cum_rev[i]

                if new_b + new_int < old_b + old_int - 1e-6:
                    seg = strips[i: j + 1]
                    seg.reverse()
                    for s in seg:
                        s.direction = -s.direction
                        s.compute_entry_exit()
                    strips[i: j + 1] = seg
                    improved = True
                    break   # restart this pass with fresh prefix sums
            if improved:
                break

    return strips


def _or_opt_1(strips: List[Strip],
              wind_vec: np.ndarray, airspeed: float,
              max_passes: int = 100,
              power_model: Optional["PowerModel"] = None,
              objective: str = "time") -> List[Strip]:
    """
    Or-opt single-strip relocation.

    For each strip at position i, evaluates inserting it at every other
    position in both orientations (forward / reversed).  Accepts the best
    improving move and restarts from the beginning.  O(r²) per pass.
    """
    n = len(strips)
    if n <= 2:
        return strips

    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1

        for i in range(n):
            s = strips[i]
            # Cost of the two edges currently incident to s
            c_in  = (_leg_cost(strips[i - 1].exit_point, s.entry_point,
                               wind_vec, airspeed, power_model, objective) if i > 0 else 0.0)
            c_out = (_leg_cost(s.exit_point, strips[i + 1].entry_point,
                               wind_vec, airspeed, power_model, objective) if i < n - 1 else 0.0)
            # Cost of the shortcut edge created when s is removed
            c_bridge = (
                _leg_cost(strips[i - 1].exit_point, strips[i + 1].entry_point,
                          wind_vec, airspeed, power_model, objective)
                if 0 < i < n - 1 else 0.0
            )
            gain_remove = c_in + c_out - c_bridge

            seq_wo = strips[:i] + strips[i + 1:]
            m = len(seq_wo)

            best_delta = 1e-9   # strict improvement threshold
            best_j     = -1
            best_dir   = s.direction

            orig_dir = s.direction
            for j in range(m + 1):
                prev_s = seq_wo[j - 1] if j > 0 else None
                next_s = seq_wo[j]     if j < m else None
                old_link = (
                    _leg_cost(prev_s.exit_point, next_s.entry_point,
                              wind_vec, airspeed, power_model, objective)
                    if (prev_s is not None and next_s is not None) else 0.0
                )
                for try_dir in (1, -1):
                    s.direction = try_dir
                    s.compute_entry_exit()
                    c_ins = (
                        (_leg_cost(prev_s.exit_point, s.entry_point,
                                   wind_vec, airspeed, power_model, objective)
                         if prev_s is not None else 0.0) +
                        (_leg_cost(s.exit_point, next_s.entry_point,
                                   wind_vec, airspeed, power_model, objective)
                         if next_s is not None else 0.0) -
                        old_link
                    )
                    delta = gain_remove - c_ins
                    if delta > best_delta:
                        best_delta = delta
                        best_j     = j
                        best_dir   = try_dir

            # Restore direction before deciding on the move
            s.direction = orig_dir
            s.compute_entry_exit()

            if best_j >= 0:
                s.direction = best_dir
                s.compute_entry_exit()
                seq_wo.insert(best_j, s)
                strips = seq_wo
                n = len(strips)
                improved = True
                break   # restart from i=0 after each accepted move

    return strips

@dataclass
class RoutingResult:
    """Output of build_route()."""
    ordered_cells: List[Cell]
    waypoints: List[np.ndarray]      # sequence of [entry, ..., exit] points
    transition_indices: List[int]    # destination waypoint index of each inter-strip transition;
                                     # segment (ti-1) is the transition leg, segment ti is a
                                     # zero-length duplicate (same point repeated)
    strips: List[Strip]
    total_time_s: float
    total_length_m: float
    turn_count: int
    total_energy_J: float = 0.0      # 0.0 when no power_model is supplied
    # Per-leg commanded airspeed and resulting ground speed.  One entry per
    # non-duplicate segment: len = len(waypoints) - 1 - len(transition_indices)
    # because each inter-strip transition inserts a zero-length duplicate waypoint
    # that carries no leg metric.  Non-empty whenever build_route() is called with
    # a power_model.  leg_airspeeds_mps[k] corresponds to segment k in the
    # waypoints list, skipping each duplicate at transition_indices[i].
    # See farmland_kml.py and planner.py for the canonical indexing pattern.
    leg_airspeeds_mps: List[float] = field(default_factory=list)
    leg_ground_speeds_mps: List[float] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)


def build_route(decomp: DecompositionResult,
                wind_vec: np.ndarray,
                airspeed: float,
                turn_threshold_deg: float = 5.0,
                power_model: Optional[PowerModel] = None,
                objective: str = "time") -> RoutingResult:
    """
    Build a wind-aware coverage route over the retained cells.

    The *objective* parameter controls what is *optimised*:

    * ``'time'``   (default) — all routing decisions minimise total flight
      time using the crab-angle ground-speed model.  If *power_model* is also
      provided, energy is computed and reported as a side metric without
      influencing any routing decision.

    * ``'energy'`` — all routing decisions (strip direction choice,
      boustrophedon tie-breaking, Or-opt-1, 2-opt) minimise total propulsion
      energy using *power_model*.  Per-leg optimal airspeeds
      ``v_a*(θ)`` are determined by ``optimal_airspeed()`` and stored in
      ``RoutingResult.leg_airspeeds_mps``; flight time is computed and
      reported using those per-leg airspeeds.  Requires *power_model*.

    When *power_model* is provided (in either mode) the output always
    includes:
      - ``total_energy_J``       — total propulsion energy in Joules
      - ``leg_airspeeds_mps``    — commanded airspeed for each leg segment
      - ``leg_ground_speeds_mps``— resulting ground speed for each leg

    These per-leg speed set-points can be programmed directly into a
    waypoint-based autopilot (e.g. ArduPlane ``DO_CHANGE_SPEED`` commands).

    Parameters
    ----------
    decomp            : DecompositionResult from decomposition.decompose()
    wind_vec          : 2-D wind vector [vwx, vwy] in m/s
    airspeed          : nominal UAV airspeed in m/s (must be > |wind_vec|)
    turn_threshold_deg: minimum heading change to count as a turn
    power_model       : PowerModel for energy computation; required when
                        objective='energy'; optional for energy reporting in
                        time mode
    objective         : 'time' or 'energy' — the routing optimisation criterion

    Returns
    -------
    RoutingResult with ordered cells, waypoints, per-leg speed set-points,
    and performance metrics.
    """
    assert airspeed > np.linalg.norm(wind_vec), (
        f"Infeasible: airspeed={airspeed} m/s ≤ wind={np.linalg.norm(wind_vec):.2f} m/s"
    )

    # ------------------------------------------------------------------
    # Step 7: Partition retained cells into strips
    # ------------------------------------------------------------------
    # Group cells by row, then find maximal contiguous column-runs
    row_cells: Dict[int, List[Cell]] = {}
    for cell in decomp.cells:
        row_cells.setdefault(cell.row, []).append(cell)
    for row in row_cells:
        row_cells[row].sort(key=lambda c: c.col)

    strips: List[Strip] = []
    strip_id = 0
    for row in sorted(row_cells.keys()):
        sorted_cells = row_cells[row]
        run: List[Cell] = [sorted_cells[0]]
        for cell in sorted_cells[1:]:
            if cell.col == run[-1].col + 1:
                run.append(cell)
            else:
                strips.append(Strip(strip_id=strip_id, row=row, cells=list(run)))
                strip_id += 1
                run = [cell]
        strips.append(Strip(strip_id=strip_id, row=row, cells=list(run)))
        strip_id += 1

    # ------------------------------------------------------------------
    # Step 8: Choose traversal direction for each strip
    # ------------------------------------------------------------------
    for strip in strips:
        ltr_entry = strip.cells[0].center
        ltr_exit = strip.cells[-1].center
        rtl_entry = strip.cells[-1].center
        rtl_exit = strip.cells[0].center

        # Estimate cost: sum of inter-cell leg costs within strip
        def strip_cost(cells_ordered: List[Cell]) -> float:
            t = 0.0
            for i in range(len(cells_ordered) - 1):
                t += _leg_cost(cells_ordered[i].center,
                               cells_ordered[i + 1].center,
                               wind_vec, airspeed, power_model, objective)
            return t

        ltr_cells = strip.cells
        rtl_cells = list(reversed(strip.cells))
        cost_ltr = strip_cost(ltr_cells)
        cost_rtl = strip_cost(rtl_cells)

        if cost_ltr < cost_rtl:
            strip.direction = 1
        elif cost_rtl < cost_ltr:
            strip.direction = -1
        else:
            # Tie-breaking: odd rows → left-to-right, even rows → right-to-left
            strip.direction = 1 if (strip.row % 2 == 1) else -1

        strip.compute_entry_exit()

    # ------------------------------------------------------------------
    # Step 9: Order strips — optimised two-stage routing
    # ------------------------------------------------------------------
    # Stage A: generate two candidate orderings
    #   (i)  Greedy nearest-exit from the bottom-left seed strip
    #   (ii) Boustrophedon (row-by-row serpentine), which avoids the
    #        "paint-yourself-into-a-corner" failure mode of pure greedy
    #        when the polygon is concave.
    # Stage B: local-search improvement
    #   Or-opt-1 (single-strip relocation, O(r²)/pass), then
    #   2-opt    (segment reversal + direction flip,  O(r²)/pass via
    #             precomputed cumulative transition sums).
    #
    # Under the "at-least-once" coverage relaxation the optimiser is free
    # to explore any ordering; every retained cell is guaranteed to be
    # visited at minimum once regardless of the strip sequence chosen.
    # ------------------------------------------------------------------

    # Make independent copies so greedy and boustrophedon don't alias
    strips_g = [_copy_strip(s) for s in strips]   # greedy candidates
    strips_b = [_copy_strip(s) for s in strips]   # boustrophedon candidates

    # --- Stage A-i: greedy nearest-exit ---
    remaining = sorted(strips_g, key=lambda s: (s.row, s.col_start))
    greedy: List[Strip] = [remaining.pop(0)]
    while remaining:
        cur_exit = greedy[-1].exit_point
        best_k, best_cost, best_flip = 0, math.inf, False
        for k, cand in enumerate(remaining):
            cf = _leg_cost(cur_exit, cand.entry_point, wind_vec, airspeed, power_model, objective)
            cr = _leg_cost(cur_exit, cand.exit_point,  wind_vec, airspeed, power_model, objective)
            if cf <= cr:
                c, flip = cf, False
            else:
                c, flip = cr, True
            if c < best_cost:
                best_cost, best_k, best_flip = c, k, flip
        chosen = remaining.pop(best_k)
        if best_flip:
            chosen.direction = -chosen.direction
            chosen.compute_entry_exit()
        greedy.append(chosen)

    # --- Stage A-ii: boustrophedon ---
    boustro = _boustrophedon_init(strips_b, wind_vec, airspeed, power_model, objective)

    # --- Pick the better initial solution ---
    cost_g = _transition_total(greedy,  wind_vec, airspeed, power_model, objective)
    cost_b = _transition_total(boustro, wind_vec, airspeed, power_model, objective)
    ordered_strips = greedy if cost_g <= cost_b else boustro

    # --- Stage B: local-search improvement ---
    ordered_strips = _or_opt_1(ordered_strips, wind_vec, airspeed,
                               power_model=power_model, objective=objective)
    ordered_strips = _two_opt(ordered_strips,  wind_vec, airspeed,
                              power_model=power_model, objective=objective)

    strips = ordered_strips

    # ------------------------------------------------------------------
    # Step 10 (tie-breaking) is subsumed by the local-search above
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Step 11: Construct the route
    # ------------------------------------------------------------------
    ordered_cells: List[Cell] = []
    waypoints: List[np.ndarray] = []
    transition_indices: List[int] = []   # destination waypoint indices of inter-strip transitions
    total_time = 0.0
    total_length = 0.0
    total_energy = 0.0
    leg_airspeeds: List[float] = []
    leg_ground_speeds: List[float] = []
    prev_exit: Optional[np.ndarray] = None

    wind_speed = float(np.linalg.norm(wind_vec))

    def _leg_metrics(p_from: np.ndarray, p_to: np.ndarray):
        """Return (time_s, energy_J, va_commanded, vg) for one leg."""
        disp = p_to - p_from
        dist = float(np.linalg.norm(disp))
        if dist < 1e-10:
            return 0.0, 0.0, airspeed, 0.0
        u = disp / dist
        w_along = float(np.dot(wind_vec, u))
        w_perp_sq = float(np.dot(wind_vec, wind_vec)) - w_along ** 2
        if objective == "energy" and power_model is not None:
            # Per-leg optimal airspeed
            theta_rad = math.atan2(math.sqrt(max(w_perp_sq, 0.0)), w_along)
            va = optimal_airspeed(theta_rad, wind_speed, power_model)
        else:
            va = airspeed
        disc = va ** 2 - w_perp_sq
        if disc <= 0:
            return math.inf, math.inf, va, 0.0
        vg = w_along + math.sqrt(disc)
        if vg <= 0:
            return math.inf, math.inf, va, 0.0
        t = dist / vg
        e = power_model.power(va) * t if power_model is not None else 0.0
        return t, e, va, vg

    for strip in strips:
        cells_in_order = strip.cells if strip.direction == 1 else list(reversed(strip.cells))

        # Transition from previous strip
        if prev_exit is not None:
            entry = strip.entry_point
            t, e, va, vg = _leg_metrics(prev_exit, entry)
            d = float(np.linalg.norm(entry - prev_exit))
            total_time += t
            total_length += d
            total_energy += e
            if power_model is not None:
                leg_airspeeds.append(va)
                leg_ground_speeds.append(vg)
            # Record the destination waypoint index (= len(waypoints) after append)
            # BEFORE appending so ti = index of entry in waypoints.
            # Segment (ti-1) is the transition; segment ti will be a zero-length
            # duplicate added by the within-strip loop below.
            transition_indices.append(len(waypoints))
            waypoints.append(entry.copy())

        # Within-strip waypoints: add entry then each subsequent cell center
        entry_added = False
        for cell in cells_in_order:
            if not entry_added:
                waypoints.append(cell.center.copy())
                entry_added = True
            else:
                t, e, va, vg = _leg_metrics(waypoints[-1], cell.center)
                d = float(np.linalg.norm(cell.center - waypoints[-1]))
                total_time += t
                total_length += d
                total_energy += e
                if power_model is not None:
                    leg_airspeeds.append(va)
                    leg_ground_speeds.append(vg)
                waypoints.append(cell.center.copy())
            ordered_cells.append(cell)

        prev_exit = strip.exit_point

    # ------------------------------------------------------------------
    # Count turns
    # ------------------------------------------------------------------
    turn_count = 0
    thresh_rad = math.radians(turn_threshold_deg)
    for i in range(1, len(waypoints) - 1):
        v1 = waypoints[i] - waypoints[i - 1]
        v2 = waypoints[i + 1] - waypoints[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            continue
        cos_a = float(np.dot(v1 / n1, v2 / n2))
        cos_a = max(-1.0, min(1.0, cos_a))
        angle = math.acos(cos_a)
        if angle > thresh_rad:
            turn_count += 1

    mean_va = (float(np.mean(leg_airspeeds)) if leg_airspeeds else airspeed)

    return RoutingResult(
        ordered_cells=ordered_cells,
        waypoints=waypoints,
        transition_indices=transition_indices,
        strips=strips,
        total_time_s=total_time,
        total_length_m=total_length,
        turn_count=turn_count,
        total_energy_J=total_energy,
        leg_airspeeds_mps=leg_airspeeds,
        leg_ground_speeds_mps=leg_ground_speeds,
        metrics={
            "n_cells": len(ordered_cells),
            "n_strips": len(strips),
            "total_time_s": total_time,
            "total_length_m": total_length,
            "turn_count": turn_count,
            "total_energy_J": total_energy,
            "mean_airspeed_mps": mean_va,
        },
    )


# ---------------------------------------------------------------------------
# Wind-optimal orientation selection
# ---------------------------------------------------------------------------

def optimal_grid_angle(
        polygon,
        cell_w: float,
        cell_h: float,
        wind_vec: np.ndarray,
        airspeed: float,
        n_angles: int = 18,
) -> float:
    """
    Return φ* ∈ [0, 180) — the grid orientation that minimises total estimated
    flight time T(φ) = coverage time + inter-strip transition time.

    A uniform sweep of *n_angles* candidate orientations (default 18, i.e. 10°
    resolution) is performed.  For each candidate φ the full decomposition and
    routing pipeline is executed and the resulting ``total_time_s`` is recorded.
    The orientation with the lowest T(φ) is returned.

    Analytical note
    ---------------
    For the coverage leg alone, round-trip time per unit length is:

        f(θ) = 1/vg(θ) + 1/vg(π−θ)
             = 2·√(va²−vw²sin²θ) / (va²−vw²)

    where θ is the angle between the strip direction and the wind vector.
    This expression is strictly minimised at θ = 90° (crosswind) for any
    wind speed vw > 0, meaning parallel-to-wind strips are never optimal
    for coverage.  Including transition time (which favours a sweep direction
    aligned with the tailwind) shifts φ* away from the exact crosswind angle
    depending on polygon geometry; the numerical sweep captures both effects.

    When wind speed is zero, T(φ) is constant and φ* = 0° is returned.

    Parameters
    ----------
    polygon   : Shapely Polygon (survey region)
    cell_w    : cell width in metres
    cell_h    : cell height in metres
    wind_vec  : horizontal wind vector [vwx, vwy] in m/s
    airspeed  : UAV airspeed (or target ground speed) in m/s; must exceed |wind_vec|
    n_angles  : number of candidate angles to evaluate (default 18 -> 10° steps)

    Returns
    -------
    phi_deg_opt : float — wind-optimal orientation in degrees, ∈ [0, 180)
    """
    if np.linalg.norm(wind_vec) < 1e-6:
        return 0.0  # no wind: all orientations equivalent, return axis-aligned

    step = 180.0 / n_angles
    best_phi, best_time = 0.0, math.inf

    for k in range(n_angles):
        phi = k * step
        decomp = _decompose_for_sweep(polygon, cell_w, cell_h, phi)
        if len(decomp.cells) == 0:
            continue
        route = build_route(decomp, wind_vec=wind_vec, airspeed=airspeed)
        if route.total_time_s < best_time:
            best_time = route.total_time_s
            best_phi = phi

    return best_phi


def optimal_grid_angle_energy(
        polygon,
        cell_w: float,
        cell_h: float,
        wind_vec: np.ndarray,
        airspeed: float,
        power_model: PowerModel,
        n_angles: int = 18,
) -> float:
    """
    Return φ*_E ∈ [0, 180) — the grid orientation that minimises total estimated
    energy consumption E(φ) under the drag-polar power model.

    Mirrors optimal_grid_angle() but passes *power_model* to build_route so that
    routing decisions (strip direction, Or-opt-1, 2-opt) use energy cost rather
    than travel time.  The sweep evaluates the same *n_angles* candidate angles.

    When wind speed is zero the energy landscape is equivalent to the time
    landscape (both are flat) and 0° is returned.

    Parameters
    ----------
    polygon      : Shapely Polygon (survey region)
    cell_w       : cell width in metres
    cell_h       : cell height in metres
    wind_vec     : horizontal wind vector [vwx, vwy] in m/s
    airspeed     : nominal airspeed (m/s) used for feasibility check; must exceed |wind_vec|
    power_model  : PowerModel instance defining P(v_a)
    n_angles     : number of candidate angles to evaluate (default 18 → 10° steps)

    Returns
    -------
    phi_deg_opt : float — energy-optimal orientation in degrees, ∈ [0, 180)
    """
    if np.linalg.norm(wind_vec) < 1e-6:
        return 0.0

    step = 180.0 / n_angles
    best_phi, best_energy = 0.0, math.inf

    for k in range(n_angles):
        phi = k * step
        decomp = _decompose_for_sweep(polygon, cell_w, cell_h, phi)
        if len(decomp.cells) == 0:
            continue
        route = build_route(decomp, wind_vec=wind_vec, airspeed=airspeed,
                            power_model=power_model, objective="energy")
        if route.total_energy_J < best_energy:
            best_energy = route.total_energy_J
            best_phi = phi

    return best_phi


# ---------------------------------------------------------------------------
# Complexity summary
# ---------------------------------------------------------------------------
# Grid generation and retention:  O(N_cand)  where N_cand = O((L/w)(L/h))
# Strip partitioning:             O(n log n) (sort by row then column)
# Direction evaluation per strip: O(n) total (constant work per cell)
# Greedy strip ordering:          O(r^2)
# Local search:                   O(K r^2) for K improving passes
# Route concatenation:            O(n)
# Overall per candidate angle:    O(N_cand + n log n + K r^2)
# where L = bounding-box side length, n = |retained cells|, r = number of strips


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from instance_gen import generate_polygon_family
    from decomposition import decompose

    poly = generate_polygon_family("mild_concave", seed=0)
    wind_dir_deg = 45.0
    wind_speed = 3.0
    airspeed = 12.0
    wind_rad = math.radians(wind_dir_deg)
    wind_vec = np.array([wind_speed * math.cos(wind_rad),
                          wind_speed * math.sin(wind_rad)])

    decomp = decompose(poly, cell_w=20.0, cell_h=20.0, angle_deg=wind_dir_deg)
    route = build_route(decomp, wind_vec=wind_vec, airspeed=airspeed)

    print(f"Cells: {route.metrics['n_cells']}")
    print(f"Strips: {route.metrics['n_strips']}")
    print(f"Flight time: {route.total_time_s:.1f} s")
    print(f"Path length: {route.total_length_m:.0f} m")
    print(f"Turn count: {route.turn_count}")
