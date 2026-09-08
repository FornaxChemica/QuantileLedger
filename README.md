# QuantileLedger

> QuantileLedger records probabilistic market forecasts before their outcomes
> are known, evaluates their calibration walk-forward, and tracks paper
> strategies without claiming an edge. Its expected result may be that the
> tested models do not outperform simple baselines; that negative result is
> part of the product, not a failure to be hidden.

**Tagline:** A point-in-time ledger for probabilistic forecasts, calibration,
and paper trades.

## What it is

A **local-first** research tool for US equities/ETFs that:

- records immutable point-in-time quantile forecasts;
- settles them with honest walk-forward rules;
- scores calibration **together with** interval sharpness;
- compares models to simple baselines on identical outcomes;
- tracks **paper-only** long-option decisions (manual and mechanical).

## What it is not

- Not a live trading system (no broker SDKs, credentials, or order routing).
- Not a claim of profitable market alpha.
- Not a cloud service (no hosted DB, CI, telemetry, or deployed dashboard).
- Not an authenticated market-data client (no API keys).

## Current status

**Milestone 0 — Foundation** is complete:

| Capability | Status |
|------------|--------|
| Package + `ql` CLI | Implemented |
| Local config (no secrets) | Implemented |
| SQLite schema + `ql init` / `ql doctor` / watchlist | Implemented |
| Offline synthetic demo | Planned (Milestone 1) |
| Baselines + calibration | Planned (Milestone 1) |
| Keyless yfinance cache | Planned (Milestone 2) |
| FinBERT / TFT | Planned (Milestones 3–4) |
| Paper ledger / mechanical | Planned (Milestones 6–7) |
| Streamlit dashboard | Planned (Milestone 8) |

There is **insufficient settled evidence** to claim forecast skill. Treat all
future live or synthetic metrics as labeled experiments.

## Quick start (Milestone 0)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11–3.12 (pinned to 3.12
via `.python-version`).

```bash
uv sync
uv run ql init
uv run ql doctor
uv run ql watch list
```

Data defaults to `./.ql/` (SQLite database, caches, logs). Override with
`--data-dir`.

Offline demo (Milestone 1, not yet available):

```bash
uv run ql demo load
uv run ql calibration
```

## Design guarantees

- **Paper only:** fills and positions are hypothetical.
- **Local only:** application data stays on your machine.
- **Keyless:** no API-key settings; optional later yfinance access is read-only
  and unofficial/research-use.
- **Point-in-time integrity:** features and news must not post-date issuance.
- **Raw forecasts are immutable;** adjusted variants are separate rows.
- **Calibration ≠ profitability:** a calibrated price forecast does not prove
  option edge.

## Evaluation notes (planned)

For P10–P90 intervals, nominal coverage is 80% (expected breach rate ~20%).
Coverage is always reported with width/sharpness and sample size `N`. Sparse
quantile CRPS is an **approximation**, not exact full-distribution CRPS.

## Privacy and artifacts

Ignored locally (never commit): `.ql/`, `*.db`, model weights, caches, reports,
and any accidental `.env` files. See `.gitignore`.

## License

MIT. Research / educational use. No warranty. Not investment advice.

## Roadmap

See [TASKS.md](TASKS.md) for the milestone checklist.
