"""Local FinBERT-style headline scoring (missing sentiment ≠ neutral).

Offline tests use FakeFinBERT (deterministic lexicon). Real ProsusAI/finbert
is optional behind ``uv sync --extra ml`` + transformers; no API key.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from quantile_ledger.errors import DatabaseError, ModelUnavailableError
from quantile_ledger.news import list_eligible_news
from quantile_ledger.timeutil import to_iso_utc, utc_now

SentimentLabel = Literal["positive", "negative", "neutral", "failed"]
ContextStatus = Literal["missing", "unscored", "scored"]

SCORER_FAKE = "fake_finbert_v1"
SCORER_FINBERT = "prosusai_finbert"
DEFAULT_FINBERT_MODEL_ID = "ProsusAI/finbert"


@dataclass(frozen=True)
class HeadlineScore:
    label: SentimentLabel
    score_positive: float
    score_negative: float
    score_neutral: float
    status: Literal["scored", "failed"]
    scorer_id: str
    model_ref: str
    is_synthetic: bool
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SentimentContext:
    """Point-in-time sentiment view for one ticker at issued_at.

    ``status='missing'`` means no eligible news — never treat as neutral.
    ``status='unscored'`` means news exists but no scorer rows yet.
    ``status='scored'`` means at least one successful score; aggregate may be
    neutral-as-scored when mean mass concentrates on the neutral class.
    """

    ticker: str
    issued_at: str
    status: ContextStatus
    n_items: int
    n_scored: int
    mean_positive: float | None
    mean_negative: float | None
    mean_neutral: float | None
    dominant_label: SentimentLabel | None
    scorer_id: str | None
    max_published_at: str | None
    max_ingested_at: str | None
    is_synthetic: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "issued_at": self.issued_at,
            "status": self.status,
            "n_items": self.n_items,
            "n_scored": self.n_scored,
            "mean_positive": self.mean_positive,
            "mean_negative": self.mean_negative,
            "mean_neutral": self.mean_neutral,
            "dominant_label": self.dominant_label,
            "scorer_id": self.scorer_id,
            "max_published_at": self.max_published_at,
            "max_ingested_at": self.max_ingested_at,
            "is_synthetic": self.is_synthetic,
            # Explicit: absence of news is not a zero/neutral feature.
            "missing_is_not_neutral": True,
        }


class HeadlineScorer(Protocol):
    scorer_id: str
    model_ref: str
    is_synthetic: bool

    def score(self, headline: str) -> HeadlineScore: ...


class FakeFinBERT:
    """Deterministic lexicon scorer for offline demo/tests (labeled synthetic)."""

    scorer_id = SCORER_FAKE
    model_ref = "fake_finbert_lexicon_v1"
    is_synthetic = True

    _POS = (
        "beat",
        "beats",
        "surge",
        "surges",
        "rally",
        "rallies",
        "upgrade",
        "upgraded",
        "record",
        "growth",
        "profit",
        "bullish",
        "outperform",
    )
    _NEG = (
        "miss",
        "misses",
        "fall",
        "falls",
        "drop",
        "drops",
        "downgrade",
        "downgraded",
        "loss",
        "lawsuit",
        "fraud",
        "bearish",
        "cut",
        "cuts",
        "plunge",
    )

    def score(self, headline: str) -> HeadlineScore:
        text = headline.lower()
        pos_hits = sum(1 for w in self._POS if w in text)
        neg_hits = sum(1 for w in self._NEG if w in text)
        if pos_hits == 0 and neg_hits == 0:
            probs = (0.15, 0.15, 0.70)
            label: SentimentLabel = "neutral"
        elif pos_hits > neg_hits:
            strength = min(0.85, 0.55 + 0.1 * pos_hits)
            rest = (1.0 - strength) / 2.0
            probs = (strength, rest, rest)
            label = "positive"
        elif neg_hits > pos_hits:
            strength = min(0.85, 0.55 + 0.1 * neg_hits)
            rest = (1.0 - strength) / 2.0
            probs = (rest, strength, rest)
            label = "negative"
        else:
            probs = (0.33, 0.33, 0.34)
            label = "neutral"
        p_pos, p_neg, p_neu = probs
        return HeadlineScore(
            label=label,
            score_positive=p_pos,
            score_negative=p_neg,
            score_neutral=p_neu,
            status="scored",
            scorer_id=self.scorer_id,
            model_ref=self.model_ref,
            is_synthetic=True,
            metadata={
                "backend": "fake_lexicon",
                "pos_hits": pos_hits,
                "neg_hits": neg_hits,
            },
        )


class TransformersFinBERT:
    """Optional ProsusAI/finbert via transformers (local weights; no API key)."""

    scorer_id = SCORER_FINBERT
    is_synthetic = False

    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        model_id: str = DEFAULT_FINBERT_MODEL_ID,
    ) -> None:
        self.model_ref = model_id
        self._model_dir = model_dir
        self._pipe: Any | None = None

    def _ensure_pipeline(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )
        except ImportError as exc:
            msg = (
                "transformers not installed. Run: uv sync --extra ml "
                "(FinBERT optional stack)"
            )
            raise ModelUnavailableError(msg) from exc
        source: str | Path = (
            self._model_dir if self._model_dir is not None else self.model_ref
        )
        if self._model_dir is not None and not self._model_dir.is_dir():
            msg = (
                f"FinBERT weights missing at {self._model_dir}. "
                "Run: ql experiment fetch-n0"
            )
            raise ModelUnavailableError(msg)
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(source))
            model = AutoModelForSequenceClassification.from_pretrained(str(source))
            self._pipe = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                top_k=None,
                function_to_apply="softmax",
            )
        except Exception as exc:
            msg = f"FinBERT load failed: {exc}"
            raise ModelUnavailableError(msg) from exc
        return self._pipe

    def score(self, headline: str) -> HeadlineScore:
        pipe = self._ensure_pipeline()
        try:
            raw = pipe(headline[:512])
        except Exception as exc:
            return HeadlineScore(
                label="failed",
                score_positive=math.nan,
                score_negative=math.nan,
                score_neutral=math.nan,
                status="failed",
                scorer_id=self.scorer_id,
                model_ref=self.model_ref,
                is_synthetic=False,
                metadata={"error": str(exc)},
            )
        # pipeline top_k=None → list[list[{label, score}]]
        rows = raw[0] if raw and isinstance(raw[0], list) else raw
        probs = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for row in rows:
            key = str(row["label"]).lower()
            if key.startswith("pos"):
                probs["positive"] = float(row["score"])
            elif key.startswith("neg"):
                probs["negative"] = float(row["score"])
            else:
                probs["neutral"] = float(row["score"])
        label = max(probs, key=probs.get)  # type: ignore[arg-type]
        return HeadlineScore(
            label=label,  # type: ignore[arg-type]
            score_positive=probs["positive"],
            score_negative=probs["negative"],
            score_neutral=probs["neutral"],
            status="scored",
            scorer_id=self.scorer_id,
            model_ref=self.model_ref,
            is_synthetic=False,
            metadata={"backend": "transformers_finbert"},
        )


def finbert_local_dir(data_dir: Path) -> Path:
    return data_dir / "models" / "finbert"


def finbert_optional_status(data_dir: Path | None = None) -> dict[str, str]:
    import importlib.util

    status = {
        "fake_finbert": "available",
        "transformers": (
            "available"
            if importlib.util.find_spec("transformers") is not None
            else "not_installed"
        ),
    }
    if data_dir is not None:
        path = finbert_local_dir(data_dir)
        status["finbert_weights"] = (
            "ready" if path.is_dir() and any(path.iterdir()) else "missing"
        )
    return status


def fetch_finbert_weights(
    data_dir: Path,
    *,
    model_id: str = DEFAULT_FINBERT_MODEL_ID,
    force: bool = False,
) -> Path:
    """One-time public Hub download of FinBERT into local .ql/ (no API key)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        msg = "huggingface_hub not installed. Run: uv sync --extra ml"
        raise ModelUnavailableError(msg) from exc
    target = finbert_local_dir(data_dir)
    if force and target.exists():
        # Leave files; snapshot_download refreshes into place.
        pass
    target.mkdir(parents=True, exist_ok=True)
    if not force and target.is_dir() and any(target.iterdir()):
        return target
    try:
        snapshot_download(repo_id=model_id, local_dir=str(target))
    except Exception as exc:
        msg = f"FinBERT download failed: {exc}"
        raise ModelUnavailableError(msg) from exc
    return target


def resolve_scorer(
    *,
    backend: Literal["fake", "finbert"] = "fake",
    data_dir: Path | None = None,
) -> HeadlineScorer:
    if backend == "fake":
        return FakeFinBERT()
    model_dir = finbert_local_dir(data_dir) if data_dir is not None else None
    return TransformersFinBERT(model_dir=model_dir)


def score_headline(scorer: HeadlineScorer, headline: str) -> HeadlineScore:
    return scorer.score(headline)


def insert_sentiment(
    conn: Any,
    *,
    news_id: str,
    score: HeadlineScore,
    scored_at: str | None = None,
) -> str:
    """Upsert one news_sentiment row; return sentiment_id."""
    import sqlite3

    sentiment_id = str(uuid.uuid4())
    scored = scored_at or to_iso_utc(utc_now())

    def _fmt(v: float) -> str | None:
        if not math.isfinite(v):
            return None
        return f"{v:.8f}"

    try:
        conn.execute(
            """
            INSERT INTO news_sentiment (
                sentiment_id, news_id, scorer_id, scored_at, label,
                score_positive, score_negative, score_neutral, status,
                is_synthetic, model_ref, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(news_id, scorer_id) DO UPDATE SET
                sentiment_id=excluded.sentiment_id,
                scored_at=excluded.scored_at,
                label=excluded.label,
                score_positive=excluded.score_positive,
                score_negative=excluded.score_negative,
                score_neutral=excluded.score_neutral,
                status=excluded.status,
                is_synthetic=excluded.is_synthetic,
                model_ref=excluded.model_ref,
                metadata_json=excluded.metadata_json
            """,
            (
                sentiment_id,
                news_id,
                score.scorer_id,
                scored,
                score.label,
                _fmt(score.score_positive),
                _fmt(score.score_negative),
                _fmt(score.score_neutral),
                score.status,
                1 if score.is_synthetic else 0,
                score.model_ref,
                json.dumps(score.metadata, sort_keys=True),
            ),
        )
    except sqlite3.Error as exc:
        msg = f"news_sentiment insert failed: {exc}"
        raise DatabaseError(msg) from exc
    return sentiment_id


def score_unscored_news(
    conn: Any,
    *,
    scorer: HeadlineScorer,
    ticker: str | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """Score news lacking a successful score for this scorer (retries failures)."""
    sql = """
        SELECT n.news_id, n.headline
        FROM news_items n
        WHERE NOT EXISTS (
            SELECT 1 FROM news_sentiment s
            WHERE s.news_id = n.news_id
              AND s.scorer_id = ?
              AND s.status = 'scored'
        )
    """
    params: list[Any] = [scorer.scorer_id]
    if ticker:
        sql += " AND n.ticker = ?"
        params.append(ticker.upper())
    sql += " ORDER BY n.published_at ASC, n.news_id ASC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    rows = conn.execute(sql, params).fetchall()
    scored = 0
    failed = 0
    for row in rows:
        result = scorer.score(str(row["headline"]))
        insert_sentiment(conn, news_id=str(row["news_id"]), score=result)
        if result.status == "scored":
            scored += 1
        else:
            failed += 1
    return {"scored": scored, "failed": failed, "considered": len(rows)}


def build_sentiment_context(
    conn: Any,
    *,
    ticker: str,
    issued_at: str,
    scorer_id: str = SCORER_FAKE,
) -> SentimentContext:
    """Aggregate PIT-eligible news scores; missing ≠ neutral."""
    items = list_eligible_news(conn, ticker=ticker, issued_at=issued_at)
    if not items:
        return SentimentContext(
            ticker=ticker.upper(),
            issued_at=issued_at,
            status="missing",
            n_items=0,
            n_scored=0,
            mean_positive=None,
            mean_negative=None,
            mean_neutral=None,
            dominant_label=None,
            scorer_id=scorer_id,
            max_published_at=None,
            max_ingested_at=None,
            is_synthetic=False,
        )

    ids = [str(r["news_id"]) for r in items]
    placeholders = ",".join("?" * len(ids))
    score_rows = conn.execute(
        f"""
        SELECT news_id, label, score_positive, score_negative, score_neutral,
               status, is_synthetic
        FROM news_sentiment
        WHERE scorer_id = ?
          AND news_id IN ({placeholders})
          AND status = 'scored'
        """,
        [scorer_id, *ids],
    ).fetchall()

    if not score_rows:
        return SentimentContext(
            ticker=ticker.upper(),
            issued_at=issued_at,
            status="unscored",
            n_items=len(items),
            n_scored=0,
            mean_positive=None,
            mean_negative=None,
            mean_neutral=None,
            dominant_label=None,
            scorer_id=scorer_id,
            max_published_at=max(str(r["published_at"]) for r in items),
            max_ingested_at=max(str(r["ingested_at"]) for r in items),
            is_synthetic=any(bool(r["is_synthetic"]) for r in items),
        )

    pos = [float(r["score_positive"]) for r in score_rows]
    neg = [float(r["score_negative"]) for r in score_rows]
    neu = [float(r["score_neutral"]) for r in score_rows]
    mean_pos = sum(pos) / len(pos)
    mean_neg = sum(neg) / len(neg)
    mean_neu = sum(neu) / len(neu)
    means = {
        "positive": mean_pos,
        "negative": mean_neg,
        "neutral": mean_neu,
    }
    dominant: SentimentLabel = max(means, key=means.get)  # type: ignore[arg-type]
    return SentimentContext(
        ticker=ticker.upper(),
        issued_at=issued_at,
        status="scored",
        n_items=len(items),
        n_scored=len(score_rows),
        mean_positive=mean_pos,
        mean_negative=mean_neg,
        mean_neutral=mean_neu,
        dominant_label=dominant,
        scorer_id=scorer_id,
        max_published_at=max(str(r["published_at"]) for r in items),
        max_ingested_at=max(str(r["ingested_at"]) for r in items),
        is_synthetic=any(bool(r["is_synthetic"]) for r in items)
        or any(bool(r["is_synthetic"]) for r in score_rows),
    )


# Back-compat stub name used by earlier milestone placeholder.
def score_headlines(
    headlines: list[str],
    *,
    backend: Literal["fake", "finbert"] = "fake",
    data_dir: Path | None = None,
) -> list[HeadlineScore]:
    scorer = resolve_scorer(backend=backend, data_dir=data_dir)
    return [scorer.score(h) for h in headlines]


__all__ = [
    "DEFAULT_FINBERT_MODEL_ID",
    "SCORER_FAKE",
    "SCORER_FINBERT",
    "ContextStatus",
    "FakeFinBERT",
    "HeadlineScore",
    "HeadlineScorer",
    "SentimentContext",
    "SentimentLabel",
    "TransformersFinBERT",
    "build_sentiment_context",
    "fetch_finbert_weights",
    "finbert_local_dir",
    "finbert_optional_status",
    "insert_sentiment",
    "resolve_scorer",
    "score_headline",
    "score_headlines",
    "score_unscored_news",
]
