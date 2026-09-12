"""Phase F: local news import + FinBERT context (missing ≠ neutral)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantile_ledger.db import SCHEMA_VERSION, connection, initialize_database
from quantile_ledger.errors import MalformedInputError
from quantile_ledger.experiments import N0
from quantile_ledger.news import (
    assert_news_pit_eligible,
    import_news_file,
    insert_news_item,
    list_eligible_news,
    news_item_from_mapping,
)
from quantile_ledger.sentiment import (
    SCORER_FAKE,
    FakeFinBERT,
    HeadlineScore,
    build_sentiment_context,
    insert_sentiment,
    score_unscored_news,
)


def test_schema_v6_news_sentiment(tmp_path: Path) -> None:
    db = tmp_path / "n.db"
    assert initialize_database(db) == SCHEMA_VERSION == 6
    with connection(db) as conn:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "news_items" in tables
    assert "news_sentiment" in tables


def test_import_and_pit_eligibility(tmp_path: Path) -> None:
    db = tmp_path / "n.db"
    initialize_database(db)
    news_path = tmp_path / "news.json"
    news_path.write_text(
        json.dumps(
            [
                {
                    "ticker": "SYN",
                    "headline": "Company beats earnings estimates",
                    "published_at": "2024-01-10T12:00:00Z",
                    "ingested_at": "2024-01-10T12:05:00Z",
                    "is_synthetic": True,
                },
                {
                    "ticker": "SYN",
                    "headline": "Late leak after issue",
                    "published_at": "2024-01-11T18:00:00Z",
                    "ingested_at": "2024-01-11T18:01:00Z",
                    "is_synthetic": True,
                },
            ]
        ),
        encoding="utf-8",
    )
    with connection(db) as conn:
        result = import_news_file(conn, news_path)
        assert result.inserted == 2
        # Idempotent
        again = import_news_file(conn, news_path)
        assert again.inserted == 0
        assert again.skipped_duplicate == 2

        issued = "2024-01-10T15:00:00Z"
        eligible = list_eligible_news(conn, ticker="SYN", issued_at=issued)
        assert len(eligible) == 1
        assert "beats" in eligible[0]["headline"]

        assert_news_pit_eligible(
            published_at="2024-01-10T12:00:00Z",
            ingested_at="2024-01-10T12:05:00Z",
            issued_at=issued,
        )
        with pytest.raises(MalformedInputError, match="leakage"):
            assert_news_pit_eligible(
                published_at="2024-01-11T18:00:00Z",
                ingested_at="2024-01-11T18:01:00Z",
                issued_at=issued,
            )


def test_missing_sentiment_is_not_neutral(tmp_path: Path) -> None:
    db = tmp_path / "n.db"
    initialize_database(db)
    with connection(db) as conn:
        missing = build_sentiment_context(
            conn,
            ticker="SYN",
            issued_at="2024-01-10T15:00:00Z",
            scorer_id=SCORER_FAKE,
        )
        assert missing.status == "missing"
        assert missing.n_items == 0
        assert missing.dominant_label is None
        assert missing.as_dict()["missing_is_not_neutral"] is True

        item = news_item_from_mapping(
            {
                "ticker": "SYN",
                "headline": "Routine update",
                "published_at": "2024-01-10T12:00:00Z",
                "ingested_at": "2024-01-10T12:01:00Z",
                "is_synthetic": True,
            }
        )
        insert_news_item(conn, item)
        unscored = build_sentiment_context(
            conn,
            ticker="SYN",
            issued_at="2024-01-10T15:00:00Z",
            scorer_id=SCORER_FAKE,
        )
        assert unscored.status == "unscored"
        assert unscored.n_items == 1
        assert unscored.mean_neutral is None


def test_fake_finbert_score_and_context(tmp_path: Path) -> None:
    db = tmp_path / "n.db"
    initialize_database(db)
    news_path = tmp_path / "news.jsonl"
    news_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ticker": "SYN",
                        "headline": "Shares surge after upgrade",
                        "published_at": "2024-01-09T10:00:00Z",
                        "ingested_at": "2024-01-09T10:05:00Z",
                        "is_synthetic": True,
                    }
                ),
                json.dumps(
                    {
                        "ticker": "SYN",
                        "headline": "Firm misses estimates, stock drops",
                        "published_at": "2024-01-09T11:00:00Z",
                        "ingested_at": "2024-01-09T11:05:00Z",
                        "is_synthetic": True,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    with connection(db) as conn:
        import_news_file(conn, news_path)
        scorer = FakeFinBERT()
        stats = score_unscored_news(conn, scorer=scorer)
        assert stats["scored"] == 2
        assert stats["failed"] == 0
        # Second pass is a no-op
        assert score_unscored_news(conn, scorer=scorer)["considered"] == 0

        ctx = build_sentiment_context(
            conn,
            ticker="SYN",
            issued_at="2024-01-10T15:00:00Z",
            scorer_id=SCORER_FAKE,
        )
        assert ctx.status == "scored"
        assert ctx.n_scored == 2
        assert ctx.dominant_label in {"positive", "negative", "neutral"}
        assert ctx.mean_positive is not None
        assert ctx.mean_negative is not None


def test_ingest_before_publish_rejected() -> None:
    with pytest.raises(MalformedInputError, match="ingested_at"):
        news_item_from_mapping(
            {
                "ticker": "SYN",
                "headline": "Bad row",
                "published_at": "2024-01-10T12:00:00Z",
                "ingested_at": "2024-01-09T12:00:00Z",
            }
        )


def test_n0_experiment_registry() -> None:
    assert N0.experiment_id == "N0"
    assert N0.feature_set == "market_plus_news"
    assert "missing" in str(N0.hyperparameters.get("missing_sentiment"))


def test_failed_sentiment_can_be_retried(tmp_path: Path) -> None:
    db = tmp_path / "n.db"
    initialize_database(db)
    item = news_item_from_mapping(
        {
            "ticker": "SYN",
            "headline": "Company beats estimates",
            "published_at": "2024-01-10T12:00:00Z",
            "ingested_at": "2024-01-10T12:01:00Z",
            "is_synthetic": True,
        }
    )
    with connection(db) as conn:
        insert_news_item(conn, item)
        insert_sentiment(
            conn,
            news_id=item.news_id,
            score=HeadlineScore(
                label="failed",
                score_positive=float("nan"),
                score_negative=float("nan"),
                score_neutral=float("nan"),
                status="failed",
                scorer_id=SCORER_FAKE,
                model_ref="test",
                is_synthetic=True,
                metadata={"error": "boom"},
            ),
        )
        stats = score_unscored_news(conn, scorer=FakeFinBERT())
        assert stats["considered"] == 1
        assert stats["scored"] == 1
        ctx = build_sentiment_context(
            conn,
            ticker="SYN",
            issued_at="2024-01-10T15:00:00Z",
            scorer_id=SCORER_FAKE,
        )
        assert ctx.status == "scored"
        assert ctx.dominant_label == "positive"


def test_pit_assert_normalizes_offset_forms() -> None:
    assert_news_pit_eligible(
        published_at="2024-01-10T12:00:00+00:00",
        ingested_at="2024-01-10T12:05:00Z",
        issued_at="2024-01-10T15:00:00+00:00",
    )
    with pytest.raises(MalformedInputError, match="leakage"):
        assert_news_pit_eligible(
            published_at="2024-01-10T16:00:00+00:00",
            ingested_at="2024-01-10T16:01:00Z",
            issued_at="2024-01-10T15:00:00Z",
        )
