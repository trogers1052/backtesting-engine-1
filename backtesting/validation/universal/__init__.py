"""Universal stock validation system."""
from .config import SECTOR_REGISTRY, SYMBOL_SECTORS, get_sector_config, get_symbol_sector
from .models import GateResult, ValidationReport, SymbolResult
from .pipeline import UniversalValidator
