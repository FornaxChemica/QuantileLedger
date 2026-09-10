# QuantileLedger

> QuantileLedger attempts to discover and convert probabilistic forecasting
> skill into paper-trading alpha while making it difficult to fool ourselves.
> Negative and inconclusive results are part of the product, not failures to hide.

**Tagline:** A point-in-time ledger for probabilistic forecasts, calibration,
    and paper trades.

## Naming note

- **Foundation Milestone 0** — package/CLI/schema bootstrap (complete).
- **Research experiment `M0`** — MambaQuantile, market-only candidate (implemented
  locally as a diagonal selective SSM + pinball; not the CUDA `mamba-ssm` package).
- **K0 / T0** — Kronos and TFT-style **challengers** compared to baseline **B1**.

These IDs are intentionally different.

## What it is

A **local-first** research tool for US equities/ETFs that:

- records immutable point-in-time quantile forecasts under a common contract;
- settles them with honest walk-forward rules;
- scores calibration **together with** interval sharpness;
- compares challenger models to simple baselines (especially **B1**) on identical outcomes;
- tracks **paper-only** trading — starting with **underlying long/flat**, with
  conservative spreads, slippage, and costs; options only after underlying
  economic-value evidence.

## What it is not

- Not a live trading system (no broker SDKs, credentials, or order routing).
- Not a promise of live profitable alpha (paper results can be negative).
- Not a cloud service (no hosted DB, CI, telemetry, or deployed dashboard).
- Not an authenticated market-data client (no API keys).

## Research matrix (current)

| ID | Candidate | Features | Status |
|----|-----------|----------|--------|
| B0 | Persistence (P50 log-return = 0) | Market only | Implemented |
| B1 | Rolling empirical return quantiles | Market only | Implemented |
| M0 | MambaQuantile (local selective SSM) | Market only | Implemented (candidate) |
| K0 | Kronos sample → empirical quantiles | Market only | Implemented (fake in demo; real via `[ml]` + `fetch-k0`) |
| T0 | TFT-style gated attention quantiles | Market only | Implemented (local; not pytorch-forecasting) |
| news / E0 | FinBERT, ensemble | — | Not implemented |

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
uv run ql experiment audit-k0
uv run ql experiment audit-t0
```

`ql demo load` uses **labeled synthetic** bars only (no network). K0 in the demo
uses `FakeKronosSampler` (not pretrained Kronos weights). T0 is a local
TFT-style model (not `pytorch-forecasting`).

For **real** Kronos (optional):

```bash
uv sync --extra ml
uv run ql experiment fetch-k0
```

That downloads public MIT weights/source once into local `.ql/` (no API key).

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
