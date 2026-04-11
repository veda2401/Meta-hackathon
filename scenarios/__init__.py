"""
scenarios/__init__.py
Power Grid Crisis Scenarios — research-grade benchmarks with predefined expected outcomes.
"""

from .scenarios import (
    SCENARIO_REGISTRY,
    get_scenario,
    list_scenarios,
    GridScenario,
)

__all__ = ["SCENARIO_REGISTRY", "get_scenario", "list_scenarios", "GridScenario"]
