# QuantileLedger task plan

Living checklist. Update status after each verified phase.

## Naming

- Foundation **Milestone 0** = package bootstrap (done).
- Research **experiment M0** = MambaQuantile market-only (Phase A).

## Foundation Milestone 0

- [x] Package, `ql`, schema, config, watchlist, tests

## Phase A — Common contract + M0

- [x] `ForecastContract` + invariant validation
- [x] Schema v2 contract columns + migration
- [x] Experiment registry B0/B1/M0
- [x] M0 local selective SSM + pinball + isotonic monotonicity
- [x] Persistence writers + `ql experiment *` / `audit-m0`
- [x] Contract / M0 tests

## Phase B — Baselines + paired comparison

- [x] B0 persistence + B1 empirical baselines
- [x] Strict paired cohort + exclusions
- [x] Pinball, coverage, width, approx CRPS, skill vs B1
- [x] `ql demo load` / `ql compare` / `ql calibration`
- [x] Hand-calculated metric + e2e tests

## Phase C — Experiment registry and reproducibility

- [x] Artifacts table + relative-path digests
- [x] Experiment freeze immutability
- [x] Auditable runs on demo load
- [x] `ql forecast inspect` provenance
- [x] `ql doctor` optional-extra status (no private paths)
- [x] Schema v3 migration tests

## Phase D — Kronos K0

- [x] Official API/license note (MIT; sample paths via `KronosPredictor`)
- [x] `K0` experiment registry + sample→quantile contract metadata
- [x] Fake adapter for offline demo/tests; real backend with local cache
- [x] Optional `[ml]` extra (torch/pandas/…) — approved
- [x] `ql experiment fetch-k0` one-time public Hub download into `.ql/`
- [x] Paired compare / calibration include K0 vs B1/M0
- [x] `ql experiment audit-k0`
## Later (not started)

- Phase E TFT T0
- Phase F FinBERT / news
- Phase G ablations
- Phase H calibration / regimes
- Phase I ensemble E0
- Phase J champion + paper
