# QuantileLedger

> QuantileLedger records probabilistic market forecasts before their outcomes
> are known, evaluates their calibration walk-forward, and tracks paper
> strategies without claiming an edge. Its expected result may be that the
> tested models do not outperform simple baselines; that negative result is
> part of the product, not a failure to be hidden.

**Tagline:** A point-in-time ledger for probabilistic forecasts, calibration,
and paper trades.

## Naming note

- **Foundation Milestone 0** — package/CLI/schema bootstrap (complete).
- **Research experiment `M0`** — MambaQuantile, market-only candidate (implemented
  locally as a diagonal selective SSM + pinball; not the CUDA `mamba-ssm` package).

These IDs are intentionally different.

## What it is

A **local-first** research tool for US equities/ETFs that:

- records immutable point-in-time quantile forecasts under a common contract;
- settles them with honest walk-forward rules;
- scores calibration **together with** interval sharpness;
- compares models to simple baselines (especially **B1**) on identical outcomes;
- tracks **paper-only** long-option decisions (manual and mechanical; later).

## What it is not

- Not a live trading system (no broker SDKs, credentials, or order routing).
- Not a claim of profitable market alpha.
- Not a cloud service (no hosted DB, CI, telemetry, or deployed dashboard).
- Not an authenticated market-data client (no API keys).

## Research matrix (current)

| ID | Candidate | Features | Status |
|----|-----------|----------|--------|
| B0 | Persistence (P50 log-return = 0) | Market only | Implemented |
| B1 | Rolling empirical return quantiles | Market only | Implemented |
| M0 | MambaQuantile (local selective SSM) | Market only | Implemented (candidate) |
| K0 / T0+ / news / E0 | Kronos, TFT, FinBERT, ensemble | — | Not implemented |

Target convention: cumulative **log return** `r = log(P_target / P_issue)`,
with price quantiles `P * exp(r)`.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11–3.12.

```bash
uv sync
uv run ql init
uv run ql doctor
uv run ql experiment list
uv run ql demo load
uv run ql compare
```

`ql demo load` uses **labeled synthetic** bars only (no network).

## Evaluation notes

For P10–P90, nominal coverage is 80%. Coverage is always shown with width and
sample size `N`. Sparse-quantile CRPS is an **approximation**. Skill vs B1:
`1 - L_model / L_B1` (positive = better than B1).

## Privacy

Ignored locally: `.ql/`, `*.db`, model weights, caches, reports. See `.gitignore`.

## License

MIT. Research / educational use. No warranty. Not investment advice.

## Roadmap

See [TASKS.md](TASKS.md).
