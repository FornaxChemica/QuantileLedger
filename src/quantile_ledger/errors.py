"""Structured domain errors for QuantileLedger."""

from __future__ import annotations


class QuantileLedgerError(Exception):
    """Base error for all domain failures."""

    def __init__(self, message: str, *, category: str = "general") -> None:
        super().__init__(message)
        self.category = category
        self.message = message


class ConfigurationError(QuantileLedgerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, category="configuration")


class ProviderError(QuantileLedgerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, category="provider")


class MalformedInputError(QuantileLedgerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, category="malformed_input")


class InsufficientDataError(QuantileLedgerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, category="insufficient_data")


class StaleDataError(QuantileLedgerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, category="stale_data")


class ModelUnavailableError(QuantileLedgerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, category="model_unavailable")


class ForecastValidationError(QuantileLedgerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, category="forecast_validation")


class DatabaseError(QuantileLedgerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, category="database")


class OutcomeUnavailableError(QuantileLedgerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, category="outcome_unavailable")


class OptionQuoteInvalidError(QuantileLedgerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, category="option_quote_invalid")


class PaperRiskRejectionError(QuantileLedgerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, category="paper_risk_rejection")


class NotImplementedMilestoneError(QuantileLedgerError):
    """Raised for CLI surfaces reserved for a later milestone."""

    def __init__(self, feature: str, milestone: str) -> None:
        super().__init__(
            f"{feature} is not implemented until {milestone}.",
            category="not_implemented",
        )
        self.feature = feature
        self.milestone = milestone
