# Task E1 Report — Synth Data Tab

**Date:** 2026-11-02  
**Branch:** worktree-research-features  
**Status:** COMPLETE

## Summary

Implemented a new "Synth Data" tab for the VegaLab research dashboard that lets researchers visualise synthetic market paths (price, ATM IV, vol smile) before running RL.

## Changes

### Backend

**`optspread/webapi.py`**
- Added `MarketConfig` import at module level.
- Added `SynthPathsOut` Pydantic model with fields: `n_days`, `n_paths`, `seed`, `spots`, `atm_ivs`, `smile_log_moneyness`, `smile_ivs`.
- Added `synth_paths(config, n_paths, seed)` function: runs `SyntheticChainProvider` for each path, captures spot/ATM IV at every step and the vol smile snapshot at `n_days // 2`.

**`optspread/server.py`**
- Added `GET /api/synth` route with query params: `sigma`, `vol_of_vol`, `skew_slope`, `alpha_bps`, `n_days`, `n_paths`, `seed`. Validates via `MarketConfig`, raises HTTP 422 on bad input.

### Frontend

**`web/lib/api.ts`**
- Added `SynthPaths` interface matching `SynthPathsOut`.
- Added `api.synthPaths(params, signal?)` method.

**`web/components/Shell.tsx`**
- Added `{ href: "/synth/", label: "Synth data" }` to TABS, between "Launch" and "Power lab".

**`web/app/synth/page.tsx`** (new file)
- "use client" component with left/right split layout.
- Left: compact form (number inputs) for sigma, vol_of_vol, skew_slope, alpha_bps, n_days, n_paths, seed + Generate button.
- Right: three ECharts panels: Price paths, ATM IV (%), Smile snapshot at mid-episode.
- Uses `--s1`–`--s5` series colors, `Chart` component, `base()`/`axis()` helpers, loading/error states.

### Tests

**`tests/test_webapi.py`**
- Added `test_synth_paths_returns_correct_shape`: verifies shape of `SynthPathsOut` for n_paths=3, n_days=10.

## Test Results

- New test: PASS
- `tests/test_webapi.py` + `tests/test_server.py`: **55 passed, 2 skipped** (pre-existing skips)
- Full test suite: exit code 0
- TypeScript (`tsc --noEmit`): clean

## Concerns

None. The `alpha_bps` parameter on `MarketConfig` has a validator `ge=0.0, le=200.0`; if a caller passes a negative value the server returns 422.
