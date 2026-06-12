# Session Notes

## 2026-06-10

- Ended with uncommitted build-vs-improve split for B/V2 work.
- New files:
  - `scripts/build_b_v2.py`
  - `tests/test_build_b_v2_script.py`
- Modified file:
  - `README.md`
- Generated scratch outputs remain untracked under `scratch/`.
- Focused verification passed:
  - `uv run pytest tests/test_build_b_v2_script.py tests/test_robust_batch.py`
  - Result: `7 passed`
- `build_b_v2.py` is intentionally separate from `improve_a_v1.py`.
  - `build_b_v2.py`: from-scratch Set B V2 seed construction plus optional column-generation rescue.
  - `improve_a_v1.py`: existing-solution benchmark polishing.
- `V2.12` known valid seeds still exist in `roadef_2016_data/hust_smart_results/`, especially `2.12_probe.xml`, official V2 ratio `0.01195`.
- From-scratch `V2.12` smoke:
  - bounded cluster constructor ran successfully but was infeasible, as expected.
  - full cluster constructor finished with `113` shifts, `841` local errors, `0` hard violations, `20` unscheduled customers.
  - one-pass rescue wiring started but was too slow for interactive work and was stopped.
- Next natural continuation:
  - either commit the build-vs-improve split,
  - or run `scripts/build_b_v2.py` longer on `V2.12` with rescue enabled to see whether it can close the remaining feasibility gap.

## 2026-06-12

- Active `V2.12` best from scratch is now:
  - `scratch/b_v2_replace_neighborhood/V2.12/best.xml`
  - local verification: `score False 4 0 108 598 0.027674488331307874`
  - remaining local errors are only `QS01` for points `124, 136, 180, 314`
  - tank violations are zero.
- Main accepted repair sequence after the prior 34-error state:
  - split 294 earlier, displace 219, re-serve 219 directly -> 27 errors.
  - reorder 56 earlier, displace 312, re-serve 312 directly -> 25 errors.
  - split 316 at 4140, displace 155 -> 13 errors.
  - HiGHS max-delivered quantity repair -> 9 errors, no tank violations.
  - multi-call-in direct route search served 170, 68, 221 -> 6 errors.
  - piggyback insertion into existing source-backed routes served 155 and 238 -> 4 errors.
- New scratch tools:
  - `scratch/multi_callin_v212.py`: bounded multi-call-in direct route generation/replacement.
  - `scratch/insert_callins_v212.py`: piggyback insertion of call-in deliveries into existing source-backed shifts.
  - `scratch/split_residual_tanks_v212.py`: tightened to reject non-QS structural errors.
- Failed/diagnostic findings:
  - Direct top-up singles from 27 did not improve.
  - Direct call-in replacement from 13 did not improve.
  - Multi-call-in rerun from 4 did not improve.
  - Remaining point 136 is resource-tight: it needs trailer 2/14, effectively driver 4 in the early window. Shift 17 currently blocks that window, and a quick reassignment search found no clean reassignment for shift 17 except its current driver/trailer.
