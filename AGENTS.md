# QuantileLedger Agent Instructions

This file defines the standing instructions for every coding agent working in
this repository, including Cursor Agent and OpenAI Codex. Read it before
planning, editing, running commands, or proposing a commit.

These instructions apply to the entire repository. A more-specific rule set may
add narrower rules for a subtree, but it must not weaken the privacy, local-
only, financial-safety, authorship, or approval requirements in this file.

- Codex reads nested `AGENTS.md` files for subtree rules.
- Cursor's nested `AGENTS.md` handling is unreliable; put subtree-specific rules
  in `.cursor/rules/*.mdc` with `globs:` so they attach only when matching files
  are edited. Keep the root file as the single source for repository-wide rules.

## Project mission

QuantileLedger is a local-first research system for recording point-in-time
probabilistic market forecasts, evaluating their walk-forward calibration, and
tracking paper-only trading decisions.

The product is the audit trail, not a claim of alpha. The expected result may
be that a model has no measurable edge. Preserve that possibility in the code,
metrics, tests, documentation, and interface.

## Instruction precedence

When instructions conflict, use this order:

1. The user's current explicit request.
2. This `AGENTS.md` and any applicable, more-specific scoped rule.
3. The repository's current architecture and tests.
4. The README, design records, and task plan.
5. General tool defaults.

Do not interpret an older plan as permission to violate a newer user decision.
If a conflict affects data integrity, privacy, authorship, Git history, or the
paper-only boundary, stop and ask the user.

## Non-negotiable boundaries

### Paper only

- Never place, route, simulate routing, or prepare a real-money order.
- Never add brokerage execution SDKs, order endpoints, broker credentials, or
  an execution-provider abstraction intended for later live trading.
- All positions, fills, marks, cash balances, and performance figures are
  hypothetical and must be labeled as paper.
- Data-provider access must be read-only.
- A manually supplied fill records a hypothetical observation only.

### Local only

- All application data, databases, caches, model artifacts, experiment logs,
  reports, and dashboards must remain on the user's machine.
- Do not add or use hosted databases, cloud storage, cloud compute, remote
  agents, hosted dashboards, analytics services, error-reporting services, or
  telemetry.
- Do not add GitHub Actions or any other hosted CI/CD workflow.
- Do not deploy the application or create cloud resources.
- Do not upload repository contents, datasets, diffs, prompts, model artifacts,
  or generated reports to third-party services.
- The only permitted remote repository write is a user-approved Git push under
  the Git protocol described below.
- Public, read-only downloads needed for local development are allowed when
  necessary: source dependencies, public model weights, public documentation,
  and keyless market data. Cache locally when appropriate.
- Before adding any dependency or feature that transmits project data, explain
  exactly what leaves the machine and obtain explicit user approval.

The IDE or model selected directly by the user may have its own processing
behavior outside this repository's control. Do not independently enable extra
cloud tools, remote execution, connectors, or data sharing.

### No API keys

- Do not request, create, read, copy, store, log, or use third-party API keys.
- Do not add API-key environment variables or secret-management infrastructure.
- Do not add `.env` examples that imply an API key is required.
- Prefer keyless public read-only sources, local file imports, bundled synthetic
  fixtures, and local models.
- If a proposed provider requires an API key, leave it unimplemented and report
  the conflict. Do not silently weaken this rule.
- Existing Git credential helpers used by the user's local Git client are not
  application API keys. Never inspect, print, export, or modify those
  credentials.

## Sources of truth

Before making a change, locate and read the relevant current sources of truth:

- `README.md` for product scope, user workflow, and public claims.
- `AGENTS.md` for working constraints.
- `pyproject.toml` and the lockfile for supported tools and dependencies.
- `src/quantile_ledger/schema.sql` and schema-version records for persistence.
- Tests for observable behavior and edge cases.
- Experiment configuration and model metadata for frozen research decisions.
- Design records, when present, for decisions that should not be rediscovered.

If a listed path does not exist yet, do not invent hidden behavior around it.
Create it only when it belongs to the current milestone.

## Working method

### Before editing

1. Inspect the repository state and applicable instructions.
2. Run `git status --short` and note existing staged, unstaged, and untracked
   files. Treat pre-existing changes as user work.
3. Read the files directly involved in the task and their nearest tests.
4. Establish the smallest coherent change that satisfies the request.
5. Identify the relevant verification commands.
6. State assumptions when they affect behavior or scientific validity.

Do not overwrite, revert, reformat, stage, or commit unrelated user changes.

### While editing

- Prefer small, reviewable patches and cohesive modules.
- Preserve public behavior unless the task intentionally changes it.
- Keep provider I/O separate from transformations and metric calculations.
- Keep financial arithmetic, time alignment, and leakage checks centralized and
  independently testable.
- Update or add tests alongside behavioral changes.
- Update documentation when assumptions, commands, configuration, schemas, or
  user-visible behavior change.
- Do not generate broad scaffolding for future phases without a current use.
- Do not add dependencies for functionality available clearly and safely in the
  standard library or existing dependencies.
- Ask before adding a new production dependency.

### When blocked

- Distinguish missing information from an implementation failure.
- Do not fabricate provider data, forecast output, test results, or successful
  command output.
- Continue with deterministic local fixtures when live keyless data is
  unavailable and the fixture path remains valid for the task.
- Stop and ask when proceeding would weaken point-in-time integrity, overwrite
  user work, change Git history, add a service, or violate a hard boundary.

## Repository search policy

Use local search tools to minimize unnecessary file reads and context usage.

### Search order

1. Use `ripgrep` (`rg`) for filenames, symbols, strings, configuration keys,
   imports, error messages, and other lexical searches.
2. Escalate to `ast-grep` only when the question depends on syntax or AST
   structure, or when a structural codemod is safer than text replacement.
3. Use semantic or embedding search only when lexical and structural search
   cannot answer the question. Do not create or upload a remote index.

Do not configure an ast-grep MCP server. The local CLI is deterministic,
requires no API key, works in both Cursor and Codex, and is sufficient for this
repository.

### ripgrep workflow

Start with filenames or matching-file lists before printing matching content:

```text
rg --files src tests
rg --files src tests | rg 'forecast|eval'
rg -l --glob '*.py' 'ForecastRecord' src tests
```

Only after narrowing to a small file set, print focused matches. Cap output to
avoid dumping noise into context:

```text
rg -n --glob '*.py' 'ForecastRecord' src/quantile_ledger/forecast -m 50
```

### ast-grep workflow

Use structural patterns for syntax-shaped questions and safe codemods, and
prefer machine-readable output for post-processing:

```text
ast-grep run -p 'def $FN($$$) -> $RET: $$$' -l python src
ast-grep run -p 'Decimal($X)' -l python src --json
```

## Simplicity standard

All else equal, prefer the simpler correct change.

- A small metric improvement does not justify opaque or brittle complexity.
- Removing code while preserving or improving correctness is a valid win.
- Do not add layers, registries, factories, or generic interfaces without two
  concrete current consumers or a clear testability need.
- Do not optimize for an arbitrary number of files.
- Avoid deeply nested package trees.
- Keep agent instructions focused; point to canonical repository files rather
  than duplicating their full contents here.

## Scientific integrity

- Never use information whose timestamp is later than forecast issuance.
- Treat raw forecasts as immutable observations.
- Store recalibrated or sentiment-adjusted output as a separate version or
  variant; never rewrite the raw forecast.
- Keep chronological training, validation, and walk-forward evaluation periods
  distinct.
- Do not tune thresholds on the final evaluation period.
- Compare complex models with simple baselines on the same settled outcomes.
- Report calibration together with interval width or another sharpness measure.
- Include sample counts with subgroup and regime metrics.
- Label quantile-based CRPS as an approximation unless a full predictive
  distribution is genuinely available.
- Do not interpret a calibrated underlying-price forecast as proof of profitable
  option trading.
- Preserve negative and inconclusive results.
- Demo and fixture data must be unmistakably labeled synthetic.

Required point-in-time assertions include, where applicable:

```text
maximum_feature_timestamp <= issued_at
training_cutoff <= issued_at
news_published_at <= issued_at
news_ingested_at <= issued_at
target_at > issued_at
```

A violation is an error, not a warning.

## Data and time rules

- Use timezone-aware timestamps internally and store canonical timestamps in
  UTC.
- Never depend on the workstation's implicit timezone.
- Use only completed market bars for forecast features and outcomes.
- Define market-session, holiday, early-close, and daylight-saving behavior in
  code and tests.
- Do not silently replace missing data with zero.
- Distinguish missing sentiment from neutral sentiment.
- Keep adjusted research prices distinct from tradable unadjusted prices.
- Record provider source, fetch time, and quality state for external data.
- Treat provider limits and schemas as runtime realities, not timeless facts.
- Cache successful keyless reads locally and make retries bounded and
  idempotent.

## Database rules

- SQLite is the local source of truth.
- Keep the initial schema and schema-version process explicit.
- Enable and test foreign-key enforcement.
- Use parameterized SQL and transactions for atomic operations.
- Use unique constraints or idempotency keys for repeatable jobs.
- Never edit a historical forecast in place.
- Preserve outcome records separately from forecast values.
- Prefer append-oriented fills and cash-ledger records for paper trading.
- Back up the database before any user-confirmed destructive reset.
- Never delete a database, model directory, or data directory through an
  unresolved variable, broad glob, recursive workspace command, or guessed
  path.

## Paper-ledger rules

- Long calls and long puts only unless the user explicitly changes the product
  scope in a later design decision.
- Use executable-side assumptions: ask for a paper purchase and bid for a paper
  sale, plus configured costs.
- Keep midpoint accounting marks distinct from bid-side liquidation marks.
- Store the quote, quote time, multiplier, fill source, slippage, fees, and data
  quality used for every paper fill.
- Never invent a fill when no valid quote exists.
- Expiration backfill must use the underlying price at the defined historical
  settlement time, not the spot when backfill happens to run.
- Reconcile cash, fills, positions, and equity in tests.
- A mechanical no-trade decision is data and must be recorded with its reason.

## Python and code style

- Use the Python version declared in `pyproject.toml`.
- Use `uv` for environments and commands unless the repository explicitly
  adopts another tool.
- Use clear type hints for public and domain-critical functions.
- Prefer small pure functions for statistical and financial calculations.
- Use `Decimal` for ledger cash and fill arithmetic; use floating point for
  model tensors and statistical computation.
- Validate at system boundaries.
- Use structured domain errors instead of returning ambiguous sentinel values.
- Avoid global mutable state and import-time network or database work.
- Keep CLI, database, provider, model, evaluation, and dashboard concerns
  separable.
- Keep calculation logic outside Streamlit rendering code.
- Use Ruff for formatting and linting and the configured type checker for static
  checks.
- Do not manually edit generated files or dependency lockfiles except through
  their owning tool.

## Testing and verification

Use the repository's declared commands. Until they are documented otherwise,
the expected local verification sequence is:

```text
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

Apply judgment for focused iterations, but run the full relevant suite before
calling a milestone complete or proposing a commit.

Tests must be deterministic and local:

- No real network calls in the default test suite.
- No API keys.
- No dependence on the user's live database, cache, home directory, clock, or
  Git configuration.
- Use temporary directories and databases.
- Fix random seeds.
- Mock provider boundaries, not core calculations.
- Include hand-calculated tests for scoring and paper P&L.
- Test idempotency, time boundaries, leakage rejection, stale data, malformed
  provider responses, and failure recovery.

Never weaken, skip, delete, or mark a test expected-to-fail merely to obtain a
green run. If a required check cannot run, report exactly why.

## Git ownership and safety

The user owns the repository, staging area, commit authorship, history, and
remote operations. The expected GitHub account is **FornaxChemica**.

### General Git rules

- Never change local or global Git identity, credential, signing, remote, or
  authentication configuration.
- Never read or expose credential-helper contents, tokens, private keys, or the
  user's full private configuration.
- Never use a bot identity, service account, alternate GitHub account, or
  agent-authored identity.
- Never use `--author` to override the user's configured identity.
- Never add `Co-authored-by`, `Generated-by`, AI, Codex, Cursor, model, or agent
  attribution to a commit.
- Respect the user's existing signing configuration. Never bypass signing or
  verification to force a commit through.
- Never run `git add`, `git add -A`, `git commit -a`, or otherwise change the
  index. The user controls staging.
- Never commit untracked or unstaged work implicitly.
- Never use destructive history or workspace commands such as hard reset,
  destructive checkout, clean, rebase, amend, filter-branch, or force push
  without a separate, explicit user request that names the operation and scope.
- Never create, change, or delete a remote, tag, release, pull request, issue,
  or branch on a remote without explicit approval.
- A commit approval is not a push approval.

### Staged-diff commit protocol

Only prepare a commit from the user's existing staged diff.

1. Run `git status --short`.
2. Inspect `git diff --cached --stat`.
3. Inspect the complete staged patch with `git diff --cached --`.
4. Run `git diff --cached --check`.
5. Confirm the staged change is coherent, contains no apparent secret, private
   dataset, local database, cache, generated model artifact, or unrelated file,
   and matches the tests being reported.
6. If nothing is staged, stop and ask the user to stage the intended files. Do
   not stage them yourself.
7. Derive the commit message exclusively from the staged diff, not from the
   broader working tree or an aspirational task description.
8. Show the user:
   - the staged file list and concise change summary;
   - relevant verification results;
   - the exact proposed commit message;
   - any remaining unstaged or untracked changes that will not be committed.
9. Ask for explicit confirmation to create that exact commit.
10. Do not interpret earlier general permission, milestone approval, or approval
    of a different diff/message as current commit approval.
11. After confirmation, recheck the staged diff. If it changed materially,
    invalidate the approval and present the new diff summary and message.
12. Commit exactly the staged content using the user's existing local Git
    identity and signing configuration.
13. If identity or signing is missing or fails, stop and ask the user to fix it.
    Do not alter Git configuration or bypass hooks.
14. Report the resulting commit hash and subject. Do not push.

Commit-message generation must happen locally from `git diff --cached`. Do not
send the diff to an API, cloud summarizer, remote model invocation, or external
service specifically for message generation.

### Conventional Commits

Every commit message must follow Conventional Commits 1.0.0:

```text
<type>[optional scope][!]: <imperative description>

[optional body explaining why]

[optional footer(s)]
```

Allowed types:

- `feat`: user-visible capability.
- `fix`: bug fix.
- `docs`: documentation only.
- `test`: tests only.
- `refactor`: internal restructure without behavior change.
- `perf`: performance improvement.
- `build`: build system or dependency change.
- `chore`: maintenance not covered above.
- `revert`: an explicitly requested revert.

Prefer stable, meaningful scopes when helpful:

```text
cli, config, data, db, forecast, sentiment, eval, paper, dashboard, docs, tests
```

Message rules:

- Use lowercase type and scope.
- Use an imperative, specific subject.
- Keep the subject concise, preferably no more than 72 characters.
- Do not end the subject with a period.
- Describe only staged behavior.
- Use a body when the reason, research constraint, migration, or tradeoff is not
  obvious from the subject.
- Mark a breaking change with `!` and/or a `BREAKING CHANGE:` footer.
- Do not invent issue numbers or claims about tests.
- Prefer one logical change per commit. If the staged diff contains unrelated
  changes, ask the user to restage it rather than forcing an inaccurate message.

Examples:

```text
feat(eval): add paired baseline skill scores
fix(paper): settle expired puts using historical spot
test(db): cover idempotent forecast insertion
docs: clarify local-only data boundaries
refactor(data): separate provider I/O from bar validation
```

### Push protocol

- Never push automatically after committing.
- Before a push, show the commit(s), current branch, and configured destination
  without exposing credentials.
- The expected GitHub repository owner is `FornaxChemica`. If the destination
  owner differs or cannot be verified, stop and ask.
- Ask for separate, explicit approval for the exact branch and remote.
- Use only the user's preconfigured Git authentication.
- Never create a token, change credentials, use an API, or switch accounts.
- Never force push.
- After pushing, report the remote and branch updated.

## Documentation rules

- Lead with what QuantileLedger measures, not with model branding.
- State that no measurable edge is an acceptable result.
- Keep synthetic, exploratory, validation, and walk-forward results visibly
  distinct.
- Document the exact price, time, fill, fee, and settlement assumptions behind
  every public metric.
- Do not write claims that exceed the available sample size or evidence.
- Keep command examples executable and local.
- Do not document cloud deployment, API-key setup, or live-order extensions.
- Update `AGENTS.md` only when a recurring, stable working rule changes, and
  obtain user approval for policy changes that affect privacy, Git, authorship,
  or project scope.

## Completion report

At the end of a task, report:

- What changed.
- Why it changed.
- Files affected.
- Tests and checks actually run, with results.
- Checks not run and why.
- Known limitations or follow-up work.
- Whether any staged, unstaged, or untracked changes remain.

Do not commit, push, deploy, or perform another external action unless the user
has separately approved it under the relevant protocol above.