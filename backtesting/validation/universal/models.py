"""Shared data models for validation results."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    data: object = None


@dataclass
class ValidationReport:
    label: str
    kwargs: Dict
    backtest: object = None  # BacktestResult
    gates: List[GateResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def pass_count(self) -> int:
        return sum(1 for g in self.gates if g.passed)

    @property
    def failed_gates(self) -> List[str]:
        return [g.name for g in self.gates if not g.passed]


@dataclass
class SymbolResult:
    """Aggregated results for one symbol."""
    symbol: str
    sector: str = ""
    individual_results: List[tuple] = field(default_factory=list)
    combo_results: List[tuple] = field(default_factory=list)
    sweep_results: List[tuple] = field(default_factory=list)
    validated_reports: List[ValidationReport] = field(default_factory=list)
    elapsed: float = 0.0
    deployed_config: Optional[Dict] = None
    recommendation: str = ""  # "deploy", "keep_current", "conditional", "skip"
