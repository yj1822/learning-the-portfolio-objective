"""Data interfaces for the simplified implementable-frontier thesis."""

from implementable_frontier.data.loaders import (
    load_cost_inputs,
    load_master_panel,
    load_risk_inputs,
)

__all__ = ["load_master_panel", "load_cost_inputs", "load_risk_inputs"]
__version__ = "0.1.0"

