"""Persist validated forecast contracts into SQLite."""

from __future__ import annotations

import json
import sqlite3

from quantile_ledger.contract import ForecastContract, validate_forecast_contract
from quantile_ledger.errors import DatabaseError, ForecastValidationError
from quantile_ledger.experiments import ExperimentSpec, get_experiment


def upsert_experiment(conn: sqlite3.Connection, spec: ExperimentSpec) -> None:
    record = spec.to_record()
    existing = conn.execute(
        "SELECT status, config_hash FROM experiments WHERE experiment_id = ?",
        (record["experiment_id"],),
    ).fetchone()
    if existing is not None and existing["status"] == "frozen":
        if existing["config_hash"] != record["config_hash"]:
            msg = (
                f"refusing to alter frozen experiment {record['experiment_id']}: "
                "config_hash mismatch"
            )
            raise DatabaseError(msg)
        return
    conn.execute(
        """
        INSERT INTO experiments (
            experiment_id, name, version, status, config_hash, label,
            started_at, min_sample_goal, notes, frozen_at
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, NULL)
        ON CONFLICT(experiment_id) DO UPDATE SET
            status = excluded.status,
            config_hash = excluded.config_hash,
            notes = excluded.notes
        WHERE experiments.status != 'frozen'
        """,
        (
            record["experiment_id"],
            record["name"],
            record["version"],
            record["status"],
            record["config_hash"],
            record["label"],
            record["purpose"],
        ),
    )


def freeze_experiment(conn: sqlite3.Connection, experiment_id: str) -> None:
    from quantile_ledger.timeutil import to_iso_utc, utc_now

    cur = conn.execute(
        """
        UPDATE experiments
        SET status = 'frozen', frozen_at = ?
        WHERE experiment_id = ?
        """,
        (to_iso_utc(utc_now()), experiment_id),
    )
    if cur.rowcount == 0:
        msg = f"experiment not found: {experiment_id}"
        raise DatabaseError(msg)


def get_forecast_provenance(
    conn: sqlite3.Connection, forecast_id: str
) -> dict[str, object]:
    row = conn.execute(
        """
        SELECT f.forecast_id, f.experiment_id, f.ticker, f.issued_at,
               f.training_cutoff, f.data_as_of, f.maximum_feature_timestamp,
               f.feature_set, f.feature_version, f.artifact_digest,
               f.model_version_id, f.target_definition, f.horizon_hours,
               e.config_hash, e.status AS experiment_status,
               m.artifact_checksum, m.family, m.hyperparameters_json
        FROM forecasts f
        LEFT JOIN experiments e ON e.experiment_id = f.experiment_id
        LEFT JOIN model_versions m ON m.model_version_id = f.model_version_id
        WHERE f.forecast_id = ?
        """,
        (forecast_id,),
    ).fetchone()
    if row is None:
        msg = f"forecast not found: {forecast_id}"
        raise DatabaseError(msg)
    return dict(row)


def ensure_model_version(
    conn: sqlite3.Connection,
    *,
    model_version_id: str,
    family: str,
    version_label: str,
    variant: str,
    training_cutoff: str,
    quantiles: list[float],
    hyperparameters: dict[str, object],
    artifact_checksum: str | None,
    seed: int | None,
    feature_schema: dict[str, object],
) -> None:
    from quantile_ledger.timeutil import to_iso_utc, utc_now

    conn.execute(
        """
        INSERT INTO model_versions (
            model_version_id, family, version_label, variant,
            artifact_path, artifact_checksum, feature_schema_json,
            hyperparameters_json, quantiles_json,
            training_start, training_end, validation_start, validation_end,
            training_cutoff, created_at, state, dependencies_json, seed,
            metrics_json, failure_notes
        ) VALUES (
            ?, ?, ?, ?, NULL, ?, ?, ?, ?, NULL, NULL, NULL, NULL,
            ?, ?, 'active', NULL, ?, NULL, NULL
        )
        ON CONFLICT(model_version_id) DO NOTHING
        """,
        (
            model_version_id,
            family,
            version_label,
            variant,
            artifact_checksum,
            json.dumps(feature_schema, sort_keys=True),
            json.dumps(hyperparameters, sort_keys=True),
            json.dumps(quantiles),
            training_cutoff,
            to_iso_utc(utc_now()),
            seed,
        ),
    )


def insert_forecast(conn: sqlite3.Connection, forecast: ForecastContract) -> str:
    """Insert a validated forecast + quantiles transactionally (caller commits)."""
    validated = validate_forecast_contract(forecast)
    try:
        spec = get_experiment(validated.experiment_id)
        upsert_experiment(conn, spec)
    except KeyError:
        # Allow ad-hoc experiment ids only if already present in DB.
        row = conn.execute(
            "SELECT 1 FROM experiments WHERE experiment_id = ?",
            (validated.experiment_id,),
        ).fetchone()
        if row is None:
            msg = f"unknown experiment_id {validated.experiment_id}; register first"
            raise DatabaseError(msg) from None
    model_version_id = validated.model_version
    ensure_model_version(
        conn,
        model_version_id=model_version_id,
        family=validated.model_family,
        version_label=validated.model_version,
        variant=validated.variant,
        training_cutoff=validated.training_cutoff,
        quantiles=validated.quantile_levels,
        hyperparameters=validated.generation_metadata,
        artifact_checksum=validated.artifact_digest,
        seed=validated.random_seed,
        feature_schema={
            "feature_set": validated.feature_set,
            "feature_version": validated.feature_version,
        },
    )
    points = validated.as_quantile_points()
    try:
        conn.execute(
            """
            INSERT INTO forecasts (
                forecast_id, run_id, experiment_id, ticker, issued_at, origin_bar_at,
                spot_at_issue, price_type, horizon_hours, target_at, model_version_id,
                variant, regime, regime_reasons_json, sentiment_context_json,
                maximum_feature_timestamp, quality, status, is_synthetic, created_at,
                feature_set, feature_version, training_cutoff, data_as_of,
                target_definition, target_transform, forecast_space,
                calibration_method, calibration_version, parent_forecast_id,
                random_seed, artifact_digest, generation_metadata_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL,
                ?, 'ok', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                validated.forecast_id,
                validated.run_id,
                validated.experiment_id,
                validated.ticker,
                validated.issued_at,
                validated.origin_bar_at,
                validated.spot_at_issue,
                validated.price_type,
                validated.nominal_horizon_hours,
                validated.target_at,
                model_version_id,
                validated.variant,
                validated.maximum_feature_timestamp,
                validated.status,
                1 if validated.is_synthetic else 0,
                validated.created_at,
                validated.feature_set,
                validated.feature_version,
                validated.training_cutoff,
                validated.data_as_of,
                validated.target_definition,
                validated.target_transform,
                validated.forecast_space,
                validated.calibration_method,
                validated.calibration_version,
                validated.parent_forecast_id,
                validated.random_seed,
                validated.artifact_digest,
                json.dumps(validated.generation_metadata, sort_keys=True),
            ),
        )
        for point in points:
            conn.execute(
                """
                INSERT INTO forecast_quantiles (
                    forecast_id, q, return_value, price_value
                )
                VALUES (?, ?, ?, ?)
                """,
                (validated.forecast_id, point.q, point.return_value, point.price_value),
            )
    except sqlite3.IntegrityError as exc:
        msg = f"forecast insert failed: {exc}"
        raise DatabaseError(msg) from exc
    except ForecastValidationError:
        raise
    return validated.forecast_id


def insert_outcome(
    conn: sqlite3.Connection,
    *,
    forecast_id: str,
    outcome_price: float,
    actual_return: float,
    outcome_bar_at: str,
    settled_at: str,
    resolution_rule: str = "first_completed_bar_at_or_after_target",
    p10: float | None = None,
    p90: float | None = None,
) -> None:
    breach = None
    if p10 is not None and p90 is not None:
        breach = 0 if p10 <= actual_return <= p90 else 1
    conn.execute(
        """
        INSERT INTO outcomes (
            forecast_id, outcome_price, outcome_bar_at, settled_at, resolution_rule,
            delay_seconds, market_bar_count, breach_p10_p90, actual_return,
            quality, unavailable_reason, error_sanitized
        ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?, 'ok', NULL, NULL)
        """,
        (
            forecast_id,
            outcome_price,
            outcome_bar_at,
            settled_at,
            resolution_rule,
            breach,
            actual_return,
        ),
    )
