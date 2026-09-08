# QuantileLedger task plan

Living checklist. Update status after each verified milestone.

## Step 0 — GitHub

- [x] Public repo `FornaxChemica/QuantileLedger` created
- [x] Local `git init` + `origin` remote
- [ ] First commit (requires staged-diff user approval)
- [ ] First push (requires separate user approval)

## Milestone 0 — Foundation

- [x] `uv` + Python 3.12 pin
- [x] `pyproject.toml` / package layout / `ql` entrypoint
- [x] MIT license, `.gitignore`, honest README
- [x] Typed non-secret config
- [x] SQLite `schema.sql` + versioning + FK/WAL
- [x] `ql init` / `ql doctor` / `ql config` / `ql watch`
- [x] Milestone stubs with nonzero exits
- [x] Verification: `ruff format --check`, `ruff check`, `mypy`, `pytest`

## Milestone 1 — Offline vertical slice

- [ ] Synthetic fixtures (labeled)
- [ ] Baselines + issuance + settlement
- [ ] Coverage, width, pinball, skill
- [ ] `ql demo load` / `ql calibration` / e2e tests

## Milestone 2 — Keyless ingestion

- [ ] yfinance optional cache + audit
- [ ] CSV/JSON/JSONL news import
- [ ] `ql data *` / doctor freshness

## Milestone 3 — FinBERT

- [ ] Local scoring + daily features + spikes
- [ ] Graceful unavailable path

## Milestone 4 — TFT

- [ ] Chronological train + registry + raw forecasts
- [ ] Skip cleanly when deps unavailable

## Milestone 5 — Evaluator hardening

- [ ] Approx CRPS/WIS, regimes, CIs, Markdown report

## Milestone 6 — Manual paper ledger

- [ ] Ask buy / bid sell / marks / expiration / reconcile

## Milestone 7 — Mechanical benchmark

- [ ] Frozen rules, no-trades, paired comparison

## Milestone 8 — Local dashboard

- [ ] Streamlit panels reading SQLite only

## Milestone 9 — Local automation / polish

- [ ] `ql nightly` lock + launchd docs + audits
