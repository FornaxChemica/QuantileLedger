# QuantileLedger task plan

Living checklist. Update status after each verified phase.

## Product claim

> QuantileLedger attempts to discover and convert probabilistic forecasting
> skill into **paper-trading alpha** while making it difficult to fool ourselves.

Pursuing alpha is intentional. Claiming live edge is not. Negative and
inconclusive results stay visible. Coverage is always reported with width and
sample size `N`.

## Naming

- Foundation **Milestone 0** = package bootstrap (done).
- Research **experiment M0** = MambaQuantile market-only (Phase A).
- **K0 / T0** = Kronos and TFT-style **challengers** vs primary baseline **B1**.

## Audit snapshot (2026-09-09)

| Area | Status |
|------|--------|
| Common log-return contract + paired compare | Done |
| B0 / B1 baselines | Done |
| M0, K0, T0 challengers (market-only) | Done (demo comparable) |
| Experiment freeze / artifacts / provenance | Done |
| Genuine forward walk-forward jobs | Partial (synthetic multi-day paper job; nightly `jobs.py` still stub) |
| Underlying long/flat paper policy | **Done** (J1: schema v5 marks + mechanical + forward-demo) |
| Policy freeze before forward paper book | Done (API + CLI + mechanical runner) |
| Forward paper P&L accumulation | Done (synthetic multi-day forward-demo + mid/bid equity curve) |
| News / FinBERT controlled ablations | Not started |
| Options paper trading | **Blocked** until underlying economic-value evidence |
| Streamlit dashboard / nightly | Not started |

## Research / trading order (authoritative)

1. Keep M0, K0, T0 comparable to **B1** under the same contract.
2. Add **underlying long/flat** paper policy (not options).
3. Use **conservative** spreads, slippage, and costs (ask buy / bid sell).
4. **Freeze** the trading policy before any forward paper book.
5. Accumulate **genuinely forward** paper results only.
6. Add news through **controlled ablations** (missing ≠ neutral).
7. Introduce **options only after** the underlying signal shows evidence of
   economic value (vs costs and vs B1).

---

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

## Phase D — Kronos K0 (challenger)

- [x] Official API/license note (MIT; sample paths via `KronosPredictor`)
- [x] `K0` experiment registry + sample→quantile contract metadata
- [x] Fake adapter for offline demo/tests; real backend with local cache
- [x] Optional `[ml]` extra (torch/pandas/…) — approved
- [x] `ql experiment fetch-k0` one-time public Hub download into `.ql/`
- [x] Paired compare / calibration include K0 vs B1/M0
- [x] `ql experiment audit-k0`

## Phase E — TFT T0 (challenger)

- [x] `T0` experiment registry (market-only, same log-return contract)
- [x] Local TFT-style gated attention + pinball + isotonic (no pytorch-forecasting dep)
- [x] Demo issues T0 alongside B0/B1/M0/K0; paired compare includes T0
- [x] `ql experiment audit-t0`
- [x] End-to-end demo/compare tests

## Phase J1 — Underlying long/flat paper (done)

Gate: paper book uses **equity long/flat only**. Options stay unimplemented.

- [x] Schema v4: `paper_policies`, `paper_decisions`, `paper_fills`, `paper_cash_ledger`
- [x] Frozen policy snapshot (`config_hash` + `frozen_at`); block trades if unfrozen
- [x] Conservative equity costs: half-spread bps, slippage bps, commission
- [x] Executable side: **ask** to go long, **bid** to flatten
- [x] Always record flat / no-trade reasons
- [x] Decimal cash + fill arithmetic; hand-calculated P&L tests
- [x] CLI: `ql paper policy-init|policy-freeze|policy-show`, account-init, history/stats
- [x] Forward flag on decisions/fills (`is_forward`)
- [x] Docs/AGENTS/TASKS/README: options deferred until underlying evidence
- [x] Mechanical runner linking settled challenger forecasts → decide/fill
- [x] Mark-to-market (mid vs bid liquidation) + equity curve report
- [x] Accumulate multi-day genuinely forward paper results (beyond unit tests)

## Phase F — Local news + FinBERT (ablations)

- [ ] Import path + point-in-time cutoffs (`published_at` / `ingested_at` ≤ `issued_at`)
- [ ] Missing sentiment ≠ neutral
- [ ] Controlled ablations only (N0 etc.); no silent feature leakage

## Phase G — Ablations T1/T2/M1/N0

- [ ] Same contract / issuance / outcome pairing as T0/M0/K0
- [ ] Coverage always with width + N

## Phase H — Calibration + regimes

- [ ] Raw forecasts immutable; recalibrated variants are separate rows
- [ ] No final-evaluation-period fitting

## Phase I — Ensemble E0

- [ ] Only after validation gates on challengers vs B1

## Phase J2 — Options paper (blocked)

Blocked until J1 forward book shows **economic value** evidence
(after costs, vs B1, with adequate N). Then:

- [ ] Long calls / long puts only (unless scope changes again)
- [ ] Option quote quality, spread caps, expiration backfill at historical spot
- [ ] Mechanical option policy on frozen champion only

## Phase K — Dashboard / nightly (later)

- [ ] Local Streamlit / reports
- [ ] launchd / local scheduler (no hosted CI)
