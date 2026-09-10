"""Paper-only underlying long/flat ledger. Never places real orders.

Options paths stay unimplemented until underlying forward results show
economic value after costs versus B1.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from quantile_ledger.errors import (
    DatabaseError,
    OptionQuoteInvalidError,
    PaperRiskRejectionError,
)
from quantile_ledger.timeutil import to_iso_utc, utc_now

Action = Literal["long", "flat"]
PolicyStatus = Literal["draft", "frozen", "retired"]

DEFAULT_POLICY_NAME = "underlying_long_flat"
DEFAULT_POLICY_VERSION = "1"


def _d(value: str | Decimal) -> Decimal:
    try:
        return value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        msg = f"invalid decimal: {value!r}"
        raise ValueError(msg) from exc


def _money(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.0001'))}"


@dataclass(frozen=True)
class UnderlyingPaperPolicy:
    """Frozen-capable equity long/flat policy with conservative costs."""

    policy_id: str
    name: str
    version: str
    status: PolicyStatus
    instrument: str
    min_p50_log_return: str
    half_spread_bps: str
    slippage_bps: str
    commission_per_share: str
    shares_per_entry: str
    max_notional_per_trade: str
    quote_max_age_seconds: int
    challenger_experiment_ids: tuple[str, ...]
    created_at: str
    frozen_at: str | None = None
    note: str | None = None

    def config_payload(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "actions": "long_flat",
            "min_p50_log_return": self.min_p50_log_return,
            "half_spread_bps": self.half_spread_bps,
            "slippage_bps": self.slippage_bps,
            "commission_per_share": self.commission_per_share,
            "shares_per_entry": self.shares_per_entry,
            "max_notional_per_trade": self.max_notional_per_trade,
            "quote_max_age_seconds": self.quote_max_age_seconds,
            "challenger_experiment_ids": list(self.challenger_experiment_ids),
        }

    def config_hash(self) -> str:
        blob = json.dumps(self.config_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def require_frozen(self) -> None:
        if self.status != "frozen":
            msg = (
                f"paper policy {self.name}/v{self.version} is {self.status}; "
                "freeze before forward testing"
            )
            raise PaperRiskRejectionError(msg)


@dataclass(frozen=True)
class EquityQuote:
    ticker: str
    bid: Decimal
    ask: Decimal
    quote_time: str
    quality: str = "ok"

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / Decimal("2")


@dataclass(frozen=True)
class PaperDecision:
    decision_id: str
    action: Action
    reason: str
    policy_hash: str
    ticker: str
    decided_at: str
    forecast_id: str | None
    experiment_id: str | None
    is_forward: bool
    is_synthetic: bool


@dataclass(frozen=True)
class PaperFill:
    fill_id: str
    side: Literal["buy", "sell"]
    quantity: Decimal
    fill_price: Decimal
    commission: Decimal
    notional: Decimal
    cash_delta: Decimal
    bid: Decimal
    ask: Decimal
    mid: Decimal
    half_spread_bps: Decimal
    slippage_bps: Decimal
    quote_time: str
    fill_time: str
    fill_source: str
    data_quality: str


def default_underlying_policy(
    *,
    policy_id: str | None = None,
    created_at: str | None = None,
) -> UnderlyingPaperPolicy:
    """Conservative default equity long/flat policy (draft until frozen)."""
    return UnderlyingPaperPolicy(
        policy_id=policy_id or str(uuid.uuid4()),
        name=DEFAULT_POLICY_NAME,
        version=DEFAULT_POLICY_VERSION,
        status="draft",
        instrument="equity",
        # Require P50 log-return above a small buffer (~10 bps) before long.
        min_p50_log_return="0.0010",
        half_spread_bps="5",
        slippage_bps="2",
        commission_per_share="0.005",
        shares_per_entry="10",
        max_notional_per_trade="5000.00",
        quote_max_age_seconds=300,
        challenger_experiment_ids=("M0", "K0", "T0"),
        created_at=created_at or to_iso_utc(utc_now()),
        frozen_at=None,
        note="Conservative underlying long/flat; options deferred.",
    )


def decide_long_flat(
    *,
    policy: UnderlyingPaperPolicy,
    ticker: str,
    p50_log_return: float,
    forecast_id: str | None = None,
    experiment_id: str | None = None,
    decided_at: str | None = None,
    is_forward: bool = True,
    is_synthetic: bool = False,
    require_frozen: bool = True,
) -> PaperDecision:
    """Map a return-space median forecast to long or flat under a frozen policy."""
    if require_frozen:
        policy.require_frozen()
    if (
        experiment_id is not None
        and experiment_id not in policy.challenger_experiment_ids
    ):
        # Baselines may still be scored; paper book only follows challengers.
        action: Action = "flat"
        reason = f"experiment_not_in_challenger_set:{experiment_id}"
    elif p50_log_return > float(_d(policy.min_p50_log_return)):
        action = "long"
        reason = (
            f"p50_log_return={p50_log_return:.6f} > min={policy.min_p50_log_return}"
        )
    else:
        action = "flat"
        reason = (
            f"p50_log_return={p50_log_return:.6f} <= min={policy.min_p50_log_return}"
        )
    return PaperDecision(
        decision_id=str(uuid.uuid4()),
        action=action,
        reason=reason,
        policy_hash=policy.config_hash(),
        ticker=ticker.upper(),
        decided_at=decided_at or to_iso_utc(utc_now()),
        forecast_id=forecast_id,
        experiment_id=experiment_id,
        is_forward=is_forward,
        is_synthetic=is_synthetic,
    )


def executable_buy_price(quote: EquityQuote, policy: UnderlyingPaperPolicy) -> Decimal:
    """Ask + slippage for a paper purchase."""
    slip = _d(policy.slippage_bps) / Decimal("10000")
    return quote.ask * (Decimal("1") + slip)


def executable_sell_price(quote: EquityQuote, policy: UnderlyingPaperPolicy) -> Decimal:
    """Bid - slippage for a paper sale / flatten."""
    slip = _d(policy.slippage_bps) / Decimal("10000")
    return quote.bid * (Decimal("1") - slip)


def build_entry_fill(
    *,
    policy: UnderlyingPaperPolicy,
    quote: EquityQuote,
    fill_source: str = "mechanical",
    fill_time: str | None = None,
) -> PaperFill:
    """Build a long entry fill at ask + slippage + commission."""
    policy.require_frozen()
    if quote.bid <= 0 or quote.ask <= 0 or quote.ask < quote.bid:
        msg = "invalid equity quote for paper entry"
        raise OptionQuoteInvalidError(msg)
    shares = _d(policy.shares_per_entry)
    price = executable_buy_price(quote, policy)
    notional = shares * price
    if notional > _d(policy.max_notional_per_trade):
        msg = (
            f"paper entry notional {notional} exceeds max "
            f"{policy.max_notional_per_trade}"
        )
        raise PaperRiskRejectionError(msg)
    commission = shares * _d(policy.commission_per_share)
    cash_delta = -(notional + commission)
    return PaperFill(
        fill_id=str(uuid.uuid4()),
        side="buy",
        quantity=shares,
        fill_price=price,
        commission=commission,
        notional=notional,
        cash_delta=cash_delta,
        bid=quote.bid,
        ask=quote.ask,
        mid=quote.mid,
        half_spread_bps=_d(policy.half_spread_bps),
        slippage_bps=_d(policy.slippage_bps),
        quote_time=quote.quote_time,
        fill_time=fill_time or to_iso_utc(utc_now()),
        fill_source=fill_source,
        data_quality=quote.quality,
    )


def build_exit_fill(
    *,
    policy: UnderlyingPaperPolicy,
    quote: EquityQuote,
    shares: Decimal,
    fill_source: str = "mechanical",
    fill_time: str | None = None,
) -> PaperFill:
    """Build a flatten fill at bid - slippage - commission."""
    policy.require_frozen()
    if shares <= 0:
        msg = "exit shares must be positive"
        raise PaperRiskRejectionError(msg)
    if quote.bid <= 0 or quote.ask <= 0 or quote.ask < quote.bid:
        msg = "invalid equity quote for paper exit"
        raise OptionQuoteInvalidError(msg)
    price = executable_sell_price(quote, policy)
    notional = shares * price
    commission = shares * _d(policy.commission_per_share)
    cash_delta = notional - commission
    return PaperFill(
        fill_id=str(uuid.uuid4()),
        side="sell",
        quantity=shares,
        fill_price=price,
        commission=commission,
        notional=notional,
        cash_delta=cash_delta,
        bid=quote.bid,
        ask=quote.ask,
        mid=quote.mid,
        half_spread_bps=_d(policy.half_spread_bps),
        slippage_bps=_d(policy.slippage_bps),
        quote_time=quote.quote_time,
        fill_time=fill_time or to_iso_utc(utc_now()),
        fill_source=fill_source,
        data_quality=quote.quality,
    )


def upsert_paper_account(
    conn: Any,
    *,
    name: str,
    starting_cash: str,
    note: str | None = None,
) -> str:
    import sqlite3

    account_id = str(uuid.uuid4())
    created = to_iso_utc(utc_now())
    try:
        conn.execute(
            """
            INSERT INTO paper_accounts (
                account_id, name, created_at, starting_cash, currency, note
            ) VALUES (?, ?, ?, ?, 'USD', ?)
            """,
            (account_id, name, created, _money(_d(starting_cash)), note),
        )
        conn.execute(
            """
            INSERT INTO paper_cash_ledger (
                entry_id, account_id, fill_id, created_at, amount,
                balance_after, kind, note
            ) VALUES (?, ?, NULL, ?, ?, ?, 'starting_cash', ?)
            """,
            (
                str(uuid.uuid4()),
                account_id,
                created,
                _money(_d(starting_cash)),
                _money(_d(starting_cash)),
                note,
            ),
        )
    except sqlite3.IntegrityError as exc:
        msg = f"paper account create failed: {exc}"
        raise DatabaseError(msg) from exc
    return account_id


def insert_policy(conn: Any, policy: UnderlyingPaperPolicy) -> None:
    import sqlite3

    try:
        conn.execute(
            """
            INSERT INTO paper_policies (
                policy_id, name, version, status, instrument, config_hash,
                config_json, created_at, frozen_at, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                policy.policy_id,
                policy.name,
                policy.version,
                policy.status,
                policy.instrument,
                policy.config_hash(),
                json.dumps(policy.config_payload(), sort_keys=True),
                policy.created_at,
                policy.frozen_at,
                policy.note,
            ),
        )
    except sqlite3.Error as exc:
        msg = f"paper policy insert failed: {exc}"
        raise DatabaseError(msg) from exc


def freeze_policy(conn: Any, policy_id: str) -> UnderlyingPaperPolicy:
    """Freeze a draft policy; thereafter config is immutable."""
    row = conn.execute(
        "SELECT * FROM paper_policies WHERE policy_id = ?",
        (policy_id,),
    ).fetchone()
    if row is None:
        msg = f"unknown paper policy_id: {policy_id}"
        raise DatabaseError(msg)
    if row["status"] == "frozen":
        return policy_from_row(row)
    if row["status"] != "draft":
        msg = f"cannot freeze paper policy in status {row['status']}"
        raise PaperRiskRejectionError(msg)
    frozen_at = to_iso_utc(utc_now())
    conn.execute(
        """
        UPDATE paper_policies
        SET status = 'frozen', frozen_at = ?
        WHERE policy_id = ? AND status = 'draft'
        """,
        (frozen_at, policy_id),
    )
    row2 = conn.execute(
        "SELECT * FROM paper_policies WHERE policy_id = ?",
        (policy_id,),
    ).fetchone()
    assert row2 is not None
    return policy_from_row(row2)


def policy_from_row(row: Any) -> UnderlyingPaperPolicy:
    cfg = json.loads(row["config_json"])
    return UnderlyingPaperPolicy(
        policy_id=row["policy_id"],
        name=row["name"],
        version=row["version"],
        status=row["status"],
        instrument=row["instrument"],
        min_p50_log_return=str(cfg["min_p50_log_return"]),
        half_spread_bps=str(cfg["half_spread_bps"]),
        slippage_bps=str(cfg["slippage_bps"]),
        commission_per_share=str(cfg["commission_per_share"]),
        shares_per_entry=str(cfg["shares_per_entry"]),
        max_notional_per_trade=str(cfg["max_notional_per_trade"]),
        quote_max_age_seconds=int(cfg["quote_max_age_seconds"]),
        challenger_experiment_ids=tuple(cfg["challenger_experiment_ids"]),
        created_at=row["created_at"],
        frozen_at=row["frozen_at"],
        note=row["note"],
    )


def get_policy(conn: Any, policy_id: str) -> UnderlyingPaperPolicy:
    row = conn.execute(
        "SELECT * FROM paper_policies WHERE policy_id = ?",
        (policy_id,),
    ).fetchone()
    if row is None:
        msg = f"unknown paper policy_id: {policy_id}"
        raise DatabaseError(msg)
    return policy_from_row(row)


def latest_cash_balance(conn: Any, account_id: str) -> Decimal:
    row = conn.execute(
        """
        SELECT balance_after FROM paper_cash_ledger
        WHERE account_id = ?
        ORDER BY created_at DESC, rowid DESC
        LIMIT 1
        """,
        (account_id,),
    ).fetchone()
    if row is None:
        msg = f"no cash ledger for account {account_id}"
        raise DatabaseError(msg)
    return _d(row["balance_after"])


def insert_decision(
    conn: Any, *, account_id: str, policy_id: str, decision: PaperDecision
) -> None:
    conn.execute(
        """
        INSERT INTO paper_decisions (
            decision_id, account_id, policy_id, policy_hash, forecast_id,
            experiment_id, ticker, decided_at, action, reason,
            is_forward, is_synthetic, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            decision.decision_id,
            account_id,
            policy_id,
            decision.policy_hash,
            decision.forecast_id,
            decision.experiment_id,
            decision.ticker,
            decision.decided_at,
            decision.action,
            decision.reason,
            1 if decision.is_forward else 0,
            1 if decision.is_synthetic else 0,
            json.dumps({}, sort_keys=True),
        ),
    )


def record_fill(
    conn: Any,
    *,
    account_id: str,
    policy: UnderlyingPaperPolicy,
    ticker: str,
    decision_id: str | None,
    fill: PaperFill,
    is_forward: bool = True,
    is_synthetic: bool = False,
) -> Decimal:
    """Append fill + cash ledger entry; return cash balance after."""
    policy.require_frozen()
    balance = latest_cash_balance(conn, account_id)
    new_balance = balance + fill.cash_delta
    if new_balance < 0:
        msg = "paper fill would make cash negative"
        raise PaperRiskRejectionError(msg)
    conn.execute(
        """
        INSERT INTO paper_fills (
            fill_id, account_id, decision_id, policy_id, ticker, side,
            quantity, fill_price, bid, ask, mid, half_spread_bps, slippage_bps,
            commission, notional, cash_delta, quote_time, fill_time, fill_source,
            data_quality, is_forward, is_synthetic, metadata_json
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            fill.fill_id,
            account_id,
            decision_id,
            policy.policy_id,
            ticker.upper(),
            fill.side,
            _money(fill.quantity),
            _money(fill.fill_price),
            _money(fill.bid),
            _money(fill.ask),
            _money(fill.mid),
            _money(fill.half_spread_bps),
            _money(fill.slippage_bps),
            _money(fill.commission),
            _money(fill.notional),
            _money(fill.cash_delta),
            fill.quote_time,
            fill.fill_time,
            fill.fill_source,
            fill.data_quality,
            1 if is_forward else 0,
            1 if is_synthetic else 0,
            json.dumps({}, sort_keys=True),
        ),
    )
    conn.execute(
        """
        INSERT INTO paper_cash_ledger (
            entry_id, account_id, fill_id, created_at, amount,
            balance_after, kind, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            account_id,
            fill.fill_id,
            fill.fill_time,
            _money(fill.cash_delta),
            _money(new_balance),
            f"fill_{fill.side}",
            None,
        ),
    )
    return new_balance


def open_share_quantity(conn: Any, account_id: str, ticker: str) -> Decimal:
    rows = conn.execute(
        """
        SELECT side, quantity FROM paper_fills
        WHERE account_id = ? AND ticker = ?
        ORDER BY fill_time, rowid
        """,
        (account_id, ticker.upper()),
    ).fetchall()
    qty = Decimal("0")
    for row in rows:
        q = _d(row["quantity"])
        if row["side"] == "buy":
            qty += q
        else:
            qty -= q
    return qty


def list_open_positions(conn: Any, account_id: str) -> dict[str, Decimal]:
    """Return tickers with positive open share quantity."""
    tickers = conn.execute(
        """
        SELECT DISTINCT ticker FROM paper_fills
        WHERE account_id = ?
        ORDER BY ticker
        """,
        (account_id,),
    ).fetchall()
    out: dict[str, Decimal] = {}
    for row in tickers:
        ticker = str(row["ticker"]).upper()
        qty = open_share_quantity(conn, account_id, ticker)
        if qty > 0:
            out[ticker] = qty
    return out


@dataclass(frozen=True)
class MarkEquity:
    cash: Decimal
    equity_mid: Decimal
    equity_bid: Decimal
    positions: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PaperMark:
    mark_id: str
    account_id: str
    marked_at: str
    cash: Decimal
    equity_mid: Decimal
    equity_bid: Decimal
    quote_source: str
    data_quality: str
    is_forward: bool
    is_synthetic: bool
    positions: tuple[dict[str, Any], ...]
    note: str | None = None


@dataclass(frozen=True)
class PaperEquitySummary:
    account_id: str
    name: str
    starting_cash: Decimal
    cash: Decimal
    equity_mid: Decimal | None
    equity_bid: Decimal | None
    mark_count: int
    decision_count: int
    forward_decision_count: int
    fill_count: int
    forward_fill_count: int
    return_mid: Decimal | None
    return_bid: Decimal | None


def compute_mark_equity(
    cash: Decimal,
    positions: dict[str, Decimal],
    quotes: dict[str, EquityQuote],
) -> MarkEquity:
    """Mid accounting equity vs bid liquidation equity (no mark-side slippage)."""
    cash = _d(cash)
    equity_mid = cash
    equity_bid = cash
    position_rows: list[dict[str, Any]] = []
    for ticker, qty in sorted(positions.items()):
        q = _d(qty)
        if q <= 0:
            continue
        quote = quotes.get(ticker.upper())
        if quote is None:
            msg = f"missing quote for open position {ticker}"
            raise PaperRiskRejectionError(msg)
        if quote.bid <= 0 or quote.ask <= 0 or quote.ask < quote.bid:
            msg = f"invalid mark quote for {ticker}"
            raise PaperRiskRejectionError(msg)
        mid = quote.mid
        bid = quote.bid
        equity_mid += q * mid
        equity_bid += q * bid
        position_rows.append(
            {
                "ticker": ticker.upper(),
                "quantity": _money(q),
                "mid": _money(mid),
                "bid": _money(bid),
                "ask": _money(quote.ask),
                "quote_time": quote.quote_time,
            }
        )
    return MarkEquity(
        cash=cash,
        equity_mid=equity_mid,
        equity_bid=equity_bid,
        positions=tuple(position_rows),
    )


def insert_mark(
    conn: Any,
    *,
    account_id: str,
    marked_at: str,
    mark: MarkEquity,
    quote_source: str,
    data_quality: str = "ok",
    is_forward: bool = True,
    is_synthetic: bool = False,
    note: str | None = None,
    mark_id: str | None = None,
) -> PaperMark:
    """Append an account-level mark snapshot (mid vs bid liquidation)."""
    mid_row = PaperMark(
        mark_id=mark_id or str(uuid.uuid4()),
        account_id=account_id,
        marked_at=marked_at,
        cash=mark.cash,
        equity_mid=mark.equity_mid,
        equity_bid=mark.equity_bid,
        quote_source=quote_source,
        data_quality=data_quality,
        is_forward=is_forward,
        is_synthetic=is_synthetic,
        positions=mark.positions,
        note=note,
    )
    conn.execute(
        """
        INSERT INTO paper_marks (
            mark_id, account_id, marked_at, cash, equity_mid, equity_bid,
            quote_source, data_quality, is_forward, is_synthetic,
            positions_json, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            mid_row.mark_id,
            mid_row.account_id,
            mid_row.marked_at,
            _money(mid_row.cash),
            _money(mid_row.equity_mid),
            _money(mid_row.equity_bid),
            mid_row.quote_source,
            mid_row.data_quality,
            1 if mid_row.is_forward else 0,
            1 if mid_row.is_synthetic else 0,
            json.dumps(list(mid_row.positions), sort_keys=True),
            mid_row.note,
        ),
    )
    return mid_row


def list_marks(conn: Any, account_id: str) -> list[PaperMark]:
    rows = conn.execute(
        """
        SELECT * FROM paper_marks
        WHERE account_id = ?
        ORDER BY marked_at ASC, rowid ASC
        """,
        (account_id,),
    ).fetchall()
    out: list[PaperMark] = []
    for row in rows:
        positions = tuple(json.loads(row["positions_json"]))
        out.append(
            PaperMark(
                mark_id=row["mark_id"],
                account_id=row["account_id"],
                marked_at=row["marked_at"],
                cash=_d(row["cash"]),
                equity_mid=_d(row["equity_mid"]),
                equity_bid=_d(row["equity_bid"]),
                quote_source=row["quote_source"],
                data_quality=row["data_quality"],
                is_forward=bool(row["is_forward"]),
                is_synthetic=bool(row["is_synthetic"]),
                positions=positions,
                note=row["note"],
            )
        )
    return out


def quotes_from_last_fills(
    conn: Any, account_id: str, tickers: list[str]
) -> dict[str, EquityQuote]:
    """Build mark quotes from the most recent fill bid/ask per ticker."""
    quotes: dict[str, EquityQuote] = {}
    for ticker in tickers:
        row = conn.execute(
            """
            SELECT bid, ask, mid, quote_time, data_quality
            FROM paper_fills
            WHERE account_id = ? AND ticker = ?
              AND bid IS NOT NULL AND ask IS NOT NULL
            ORDER BY fill_time DESC, rowid DESC
            LIMIT 1
            """,
            (account_id, ticker.upper()),
        ).fetchone()
        if row is None:
            msg = f"no fill quote available to mark {ticker}"
            raise PaperRiskRejectionError(msg)
        quotes[ticker.upper()] = EquityQuote(
            ticker=ticker.upper(),
            bid=_d(row["bid"]),
            ask=_d(row["ask"]),
            quote_time=row["quote_time"] or "",
            quality=row["data_quality"] or "ok",
        )
    return quotes


def record_account_mark(
    conn: Any,
    *,
    account_id: str,
    quotes: dict[str, EquityQuote],
    marked_at: str | None = None,
    quote_source: str,
    data_quality: str = "ok",
    is_forward: bool = True,
    is_synthetic: bool = False,
    note: str | None = None,
) -> PaperMark:
    """Mark open positions at provided quotes; cash-only books are allowed."""
    cash = latest_cash_balance(conn, account_id)
    positions = list_open_positions(conn, account_id)
    mark = compute_mark_equity(cash, positions, quotes)
    return insert_mark(
        conn,
        account_id=account_id,
        marked_at=marked_at or to_iso_utc(utc_now()),
        mark=mark,
        quote_source=quote_source,
        data_quality=data_quality,
        is_forward=is_forward,
        is_synthetic=is_synthetic,
        note=note,
    )


def paper_equity_summary(conn: Any, account_id: str) -> PaperEquitySummary:
    acct = conn.execute(
        """
        SELECT account_id, name, starting_cash FROM paper_accounts
        WHERE account_id = ?
        """,
        (account_id,),
    ).fetchone()
    if acct is None:
        msg = f"unknown paper account_id: {account_id}"
        raise DatabaseError(msg)
    starting = _d(acct["starting_cash"])
    cash = latest_cash_balance(conn, account_id)
    marks = list_marks(conn, account_id)
    equity_mid: Decimal | None = None
    equity_bid: Decimal | None = None
    return_mid: Decimal | None = None
    return_bid: Decimal | None = None
    if marks:
        equity_mid = marks[-1].equity_mid
        equity_bid = marks[-1].equity_bid
        if starting != 0:
            return_mid = (equity_mid - starting) / starting
            return_bid = (equity_bid - starting) / starting
    decision_count = int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM paper_decisions WHERE account_id = ?",
            (account_id,),
        ).fetchone()["n"]
    )
    forward_decision_count = int(
        conn.execute(
            """
            SELECT COUNT(*) AS n FROM paper_decisions
            WHERE account_id = ? AND is_forward = 1
            """,
            (account_id,),
        ).fetchone()["n"]
    )
    fill_count = int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM paper_fills WHERE account_id = ?",
            (account_id,),
        ).fetchone()["n"]
    )
    forward_fill_count = int(
        conn.execute(
            """
            SELECT COUNT(*) AS n FROM paper_fills
            WHERE account_id = ? AND is_forward = 1
            """,
            (account_id,),
        ).fetchone()["n"]
    )
    return PaperEquitySummary(
        account_id=account_id,
        name=acct["name"],
        starting_cash=starting,
        cash=cash,
        equity_mid=equity_mid,
        equity_bid=equity_bid,
        mark_count=len(marks),
        decision_count=decision_count,
        forward_decision_count=forward_decision_count,
        fill_count=fill_count,
        forward_fill_count=forward_fill_count,
        return_mid=return_mid,
        return_bid=return_bid,
    )


# Avoid exporting broken apply_fill.
__all__ = [
    "EquityQuote",
    "MarkEquity",
    "PaperDecision",
    "PaperEquitySummary",
    "PaperFill",
    "PaperMark",
    "UnderlyingPaperPolicy",
    "build_entry_fill",
    "build_exit_fill",
    "compute_mark_equity",
    "decide_long_flat",
    "default_underlying_policy",
    "executable_buy_price",
    "executable_sell_price",
    "freeze_policy",
    "get_policy",
    "insert_decision",
    "insert_mark",
    "insert_policy",
    "latest_cash_balance",
    "list_marks",
    "list_open_positions",
    "open_share_quantity",
    "paper_equity_summary",
    "policy_from_row",
    "quotes_from_last_fills",
    "record_account_mark",
    "record_fill",
    "upsert_paper_account",
]
