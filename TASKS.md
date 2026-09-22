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

## Audit snapshot (2026-09-21)

| Area | Status |
|------|--------|
| Common log-return contract + paired compare | Done |
| B0 / B1 baselines | Done |
| M0, K0, T0 challengers (market-only) | Done (demo comparable) |
| Experiment freeze / artifacts / provenance | Done |
| Bars ingest (local import + Stooq daily fetch) | **Done** (Phase WF / Milestone 2) |
| Genuine forward walk-forward on stored bars | **Done** (`ql experiment walk-forward`; nightly still stub) |
| Evidence gate report vs B1 (skill + paper) | **Done** (`ql report gate`; human review) |
| Underlying long/flat paper policy | **Done** (J1: schema v5 marks + mechanical + forward-demo) |
| Policy freeze before forward paper book | Done (API + CLI + mechanical runner) |
| Forward paper P&L accumulation | Done (synthetic demo + bars gate books) |
| News / FinBERT controlled ablations | **Done** (Phase F infra + Phase G T1/T2/M1/N0 ladder) |
| Isotonic PIT recalibration + vol regimes | **Done** (Phase H: M0c/T0c/K0c, val/eval split) |
| Ensemble E0 | **Blocked** until gate shows skill vs B1 with adequate N |
| Options paper trading | **Blocked** until gate shows paper equity delta vs B1 |
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

- [x] Import path + point-in-time cutoffs (`published_at` / `ingested_at` ≤ `issued_at`)
- [x] Missing sentiment ≠ neutral
- [x] Controlled ablations only (N0 etc.); no silent feature leakage

## Phase G — Ablations T1/T2/M1/N0

- [x] Same contract / issuance / outcome pairing as T0/M0/K0
- [x] Coverage always with width + N
- [x] Wire N0 news context into challenger forecast variants (not silent merge)

Ladder: **T1** = T0 + news volume; **T2** = T0 + volume + FinBERT; **M1** = M0 + volume + FinBERT; **N0** = news-only baseline vs B1. Missing/partial context refuses issuance.

## Phase H — Calibration + regimes

- [x] Raw forecasts immutable; recalibrated variants are separate rows
- [x] No final-evaluation-period fitting

`M0c`/`T0c`/`K0c` = isotonic PIT children (`variant=isotonic_v1`) of M0/T0/K0.
Fit on validation `issued_at` only; apply on eval only. Regimes = PIT realized-vol
terciles with N + coverage + width.

## Phase WF — Genuine walk-forward evidence gate (Milestone 2)

- [x] `bars` upsert + PIT `list_closes_as_of`
- [x] `ql data import-bars` (CSV/JSON/JSONL; `--synthetic` for fixtures)
- [x] Keyless Stooq daily fetch + `.ql/cache/bars/` (`ql data fetch`)
- [x] Expanded `ql data status` (bars + news)
- [x] `ql experiment walk-forward` on stored `1d` bars (horizons 24h / 72h)
- [x] B1 control paper policy (`ql paper policy-init --b1-control`) + `ql paper gate-run`
- [x] `ql report gate` (excludes synthetic; pass/fail/inconclusive; human unlock checklist)
- [ ] Nightly scheduler / `jobs.py` (still Milestone 9)

Gate does **not** auto-unlock Phase I or J2.

## Phase I — Ensemble E0

- [ ] Only after validation gates on challengers vs B1 (`ql report gate` non-inconclusive + positive skill; human review)

## Phase J2 — Options paper (blocked)

Blocked until J1 forward book shows **economic value** evidence
(after costs, vs B1, with adequate N) via `ql report gate` paper delta. Then:

- [ ] Long calls / long puts only (unless scope changes again)
- [ ] Option quote quality, spread caps, expiration backfill at historical spot
- [ ] Mechanical option policy on frozen champion only

## Phase K — Dashboard / nightly (later)

- [ ] Local Streamlit / reports
- [ ] launchd / local scheduler (no hosted CI)
