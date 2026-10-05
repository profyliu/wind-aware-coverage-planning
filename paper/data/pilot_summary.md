# Pilot Study Summary — Scenario 01

**Date:** 2026-04-07  
**Polygon:** Mildly-concave 12-vertex, ~500 m × 400 m bounding box  
**Cell size:** 20 m × 20 m  
**Wind:** 3.0 m/s at 45° (NE); airspeed 12.0 m/s (wind ratio 0.25)

---

## Key Results

| Method | Cells | Boundary cells | Coverage | Flight time (s) | Path length (m) | Turns | Runtime (ms) |
|---|---|---|---|---|---|---|---|
| Proposed (wind-aligned, selective) | 528 | 113 | 1.000000 | 1774 | 19 965 | 28 | 36.9 |
| Baseline 1 (fixed 0°, selective) | 532 | 98 | 1.000000 | 1735 | 20 312 | 23 | 29.3 |
| Baseline 2 (wind-aligned, non-pruned) | 930 | 113 | 1.000000 | 3134 | 35 410 | 29 | 46.5 |

---

## Observations

### Coverage completeness
- **Coverage ratio = 1.000000 for all three methods.**  Lemma 1 (any-intersection rule
  implies P ⊆ ∪ retained cells) is confirmed empirically on the canonical polygon.
- No sliver gaps or uncovered boundary segments were detected.

### Selective vs. non-pruned grid (Baseline 2)
- The proposed selective grid retains **528 cells vs. 930** for the non-pruned bounding-box
  grid — a 43% reduction in cell count, directly reducing flight time from 3 134 s to 1 774 s
  (−43%).  This confirms that selective (any-intersection) pruning is essential for efficiency.

### Wind-aligned vs. fixed orientation (Baseline 1)
- At the specific test wind of 45° NE the proposed (wind-aligned) method produces slightly
  *longer* flight time (1 774 s) than the fixed-0° baseline (1 735 s), a difference of +2.3%.
- This result is expected and honest: aligning the grid with 45° NE does not guarantee a
  shorter flight time at that particular wind direction for this polygon geometry.
- The wind-sweep experiment (see `wind_sweep_scenario01.csv`) shows that the wind-aligned
  method outperforms the fixed baseline at some wind directions (e.g., 180° by +3.8%) but
  underperforms at others (e.g., 270° by −78%).  This pattern is a key finding to discuss
  in the manuscript: benefits are direction-dependent and geometry-dependent.

### Wind direction sweep
| Wind dir (°) | Proposed (s) | Baseline 1 (s) | Δ (s) |
|---|---|---|---|
| 0 | 1802 | 1802 | 0 |
| 45 | 1774 | 1735 | −39 |
| 90 | 1729 | 962 | −767 |
| 135 | 1765 | 1748 | −17 |
| 180 | 1751 | 1817 | +66 |
| 225 | 1778 | 1768 | −10 |
| 270 | 1738 | 982 | −757 |
| 315 | 1770 | 1755 | −15 |

- The large negative deltas at 90° and 270° arise because Baseline 1 (fixed 0° grid) has
  strips running E–W; at N–S wind the fixed baseline generates very short strips aligned
  favourably with that wind, while the proposed method aligns strips at 90°/270°.
  This is a geometry artefact of this specific polygon and warrants further investigation.

### Planner runtime
- All planners complete in < 50 ms for the 500 m × 500 m polygon at 20 m cells.
- Runtimes are consistent with the O(N_cand + n log n) complexity bound.

---

## Failure modes observed
- None.  All instances returned valid connected paths with coverage = 1.000000.
- The `verify_coverage()` assertion passed for all 576 pilot experiments.

---

## Next steps for manuscript integration
1. Replace the prospective language in `sec_experiments.tex` with the table values above
   for Scenario 01.
2. Add a note in the Analysis section referencing the empirical coverage confirmation.
3. Discuss the wind-direction-dependent performance pattern in the results narrative.
4. Produce full-scale results with N_SEEDS=20 (`python experiments.py --seeds 20`) and
   update tables with averaged statistics.
