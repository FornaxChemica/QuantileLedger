-- QuantileLedger schema v2
-- Authoritative local SQLite schema. Apply via ql init / db.initialize_database.
-- Paper-only / local-only product: no live-order concepts.
-- v2 adds common forecast-contract columns for multi-model comparison.
-- Research experiment M0 (MambaQuantile) is distinct from foundation Milestone 0.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS config_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    config_json TEXT NOT NULL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS watchlist (
    ticker TEXT PRIMARY KEY,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    added_at TEXT NOT NULL,
    removed_at TEXT,
    exchange TEXT,
    currency TEXT,
    note TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    run_type TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL,
    provider TEXT,
    software_version TEXT,
    config_snapshot_id TEXT REFERENCES config_snapshots (snapshot_id),
    row_counts_json TEXT,
    error_sanitized TEXT
);

CREATE TABLE IF NOT EXISTS run_ticker_results (
    run_id TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    status TEXT NOT NULL,
    rows_affected INTEGER NOT NULL DEFAULT 0,
    retries INTEGER NOT NULL DEFAULT 0,
    error_sanitized TEXT,
    PRIMARY KEY (run_id, ticker)
);

CREATE TABLE IF NOT EXISTS model_versions (
    model_version_id TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    version_label TEXT NOT NULL,
    variant TEXT NOT NULL DEFAULT 'raw',
    artifact_path TEXT,
    artifact_checksum TEXT,
    feature_schema_json TEXT,
    hyperparameters_json TEXT,
    quantiles_json TEXT,
    training_start TEXT,
    training_end TEXT,
    validation_start TEXT,
    validation_end TEXT,
    training_cutoff TEXT,
    created_at TEXT NOT NULL,
    state TEXT NOT NULL,
    dependencies_json TEXT,
    seed INTEGER,
    metrics_json TEXT,
    failure_notes TEXT,
    UNIQUE (family, version_label, variant)
);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES runs (run_id),
    experiment_id TEXT REFERENCES experiments (experiment_id),
    ticker TEXT NOT NULL,
    issued_at TEXT NOT NULL,
    origin_bar_at TEXT NOT NULL,
    spot_at_issue REAL NOT NULL CHECK (spot_at_issue > 0),
    price_type TEXT NOT NULL,
    horizon_hours INTEGER NOT NULL CHECK (horizon_hours > 0),
    target_at TEXT NOT NULL,
    model_version_id TEXT NOT NULL REFERENCES model_versions (model_version_id),
    variant TEXT NOT NULL DEFAULT 'raw',
    regime TEXT,
    regime_reasons_json TEXT,
    sentiment_context_json TEXT,
    maximum_feature_timestamp TEXT NOT NULL,
    quality TEXT NOT NULL DEFAULT 'ok',
    status TEXT NOT NULL DEFAULT 'issued',
    is_synthetic INTEGER NOT NULL DEFAULT 0 CHECK (is_synthetic IN (0, 1)),
    created_at TEXT NOT NULL,
    feature_set TEXT,
    feature_version TEXT,
    training_cutoff TEXT,
    data_as_of TEXT,
    target_definition TEXT,
    target_transform TEXT,
    forecast_space TEXT,
    calibration_method TEXT,
    calibration_version TEXT,
    parent_forecast_id TEXT,
    random_seed INTEGER,
    artifact_digest TEXT,
    generation_metadata_json TEXT,
    CHECK (target_at > issued_at),
    CHECK (maximum_feature_timestamp <= issued_at),
    UNIQUE (ticker, issued_at, horizon_hours, model_version_id, variant)
);

CREATE TABLE IF NOT EXISTS forecast_quantiles (
    forecast_id TEXT NOT NULL REFERENCES forecasts (forecast_id) ON DELETE CASCADE,
    q REAL NOT NULL CHECK (q > 0 AND q < 1),
    return_value REAL NOT NULL,
    price_value REAL NOT NULL CHECK (price_value > 0),
    PRIMARY KEY (forecast_id, q)
);

CREATE TABLE IF NOT EXISTS outcomes (
    forecast_id TEXT PRIMARY KEY REFERENCES forecasts (forecast_id),
    outcome_price REAL,
    outcome_bar_at TEXT,
    settled_at TEXT NOT NULL,
    resolution_rule TEXT NOT NULL,
    delay_seconds REAL,
    market_bar_count INTEGER,
    breach_p10_p90 INTEGER CHECK (breach_p10_p90 IN (0, 1)),
    actual_return REAL,
    quality TEXT NOT NULL DEFAULT 'ok',
    unavailable_reason TEXT,
    error_sanitized TEXT
);

CREATE TABLE IF NOT EXISTS metric_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    evaluated_at TEXT NOT NULL,
    window_start TEXT,
    window_end TEXT,
    model_version_id TEXT REFERENCES model_versions (model_version_id),
    variant TEXT,
    horizon_hours INTEGER,
    ticker TEXT,
    regime TEXT,
    sample_count INTEGER NOT NULL CHECK (sample_count >= 0),
    metrics_json TEXT NOT NULL,
    confidence_json TEXT,
    baseline_reference TEXT,
    is_synthetic INTEGER NOT NULL DEFAULT 0 CHECK (is_synthetic IN (0, 1))
);

-- Stub shells for later milestones (safe empty tables).
CREATE TABLE IF NOT EXISTS bars (
    bar_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    interval TEXT NOT NULL,
    bar_start TEXT NOT NULL,
    bar_end TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL NOT NULL,
    volume REAL,
    adjustment TEXT NOT NULL,
    provider TEXT NOT NULL,
    first_fetched_at TEXT NOT NULL,
    last_observed_at TEXT NOT NULL,
    revision_state TEXT NOT NULL DEFAULT 'initial',
    quality TEXT NOT NULL DEFAULT 'ok',
    is_synthetic INTEGER NOT NULL DEFAULT 0 CHECK (is_synthetic IN (0, 1)),
    UNIQUE (provider, ticker, interval, bar_end)
);

CREATE TABLE IF NOT EXISTS news_items (
    news_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    headline TEXT NOT NULL,
    source TEXT,
    url TEXT,
    published_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    dedupe_key TEXT NOT NULL,
    is_retrospective INTEGER NOT NULL DEFAULT 0 CHECK (is_retrospective IN (0, 1)),
    quality TEXT NOT NULL DEFAULT 'ok',
    is_synthetic INTEGER NOT NULL DEFAULT 0 CHECK (is_synthetic IN (0, 1)),
    UNIQUE (dedupe_key)
);

CREATE TABLE IF NOT EXISTS paper_accounts (
    account_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    starting_cash TEXT NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD',
    note TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    status TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT 'exploratory',
    started_at TEXT,
    min_sample_goal INTEGER,
    notes TEXT,
    frozen_at TEXT,
    UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    digest TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    experiment_id TEXT REFERENCES experiments (experiment_id),
    model_version_id TEXT REFERENCES model_versions (model_version_id),
    metadata_json TEXT
);
