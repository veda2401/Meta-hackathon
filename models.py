"""
models.py
---------
Pydantic action / observation / state models for the Power Grid OpenEnv client.
Mirrors the structure of carla_env/models.py, calendar_env/models.py, etc.

Usage
-----
    from power_grid import PowerGridAction, PowerGridEnv
    async with PowerGridEnv(base_url="http://localhost:7860") as env:
        result = await env.reset(difficulty="hard", scenario="cascade_blackout")
        result = await env.step(PowerGridAction(dispatch_mw=[80,60,40,45,35,25]))
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Action
# ─────────────────────────────────────────────────────────────────────────────

class PowerGridAction(BaseModel):
    """Action sent to the environment at each step.

    Fields
    ------
    dispatch_mw : list of 6 floats
        MW setpoints for generators 0-5.
        Gen 0 (Coal, Bus 1): 20-100 MW
        Gen 1 (Gas,  Bus 2): 10-80 MW
        Gen 2 (Gas,  Bus 3): 5-50 MW
        Gen 3 (Wind, Bus 6): 0-50 MW  (weather-limited)
        Gen 4 (Hydro,Bus 8): 5-40 MW
        Gen 5 (Solar,Bus12): 0-30 MW  (weather-limited)
    """
    dispatch_mw: List[float] = Field(
        default=[80.0, 60.0, 40.0, 45.0, 35.0, 25.0],
        description="MW setpoints for 6 generators",
        min_length=6,
        max_length=6,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Observation  (returned after reset() and step())
# ─────────────────────────────────────────────────────────────────────────────

class PowerGridObservation(BaseModel):
    """Full environment observation after reset or step.

    Physics fields
    --------------
    bus_angles_deg       : voltage angles at all 14 buses (degrees)
    line_flows_mw        : signed DC power flows on all 20 lines (MW)
    line_loading_frac    : |flow| / rating per line  (1.0 = at capacity)
    line_status          : True = online, False = relay-tripped
    gen_dispatch_mw      : actual generator output after dispatch (MW)
    gen_available_mw     : maximum currently available output (weather-limited)
    gen_online           : True = online, False = forced outage
    bus_loads_mw         : stochastic load at each bus (MW)
    total_load_mw        : sum of all bus loads
    total_gen_mw         : sum of all generator outputs
    power_balance_mw     : generation minus load (MW); ideal = 0
    cf_wind              : wind capacity factor [0,1]
    cf_solar             : solar capacity factor [0,1]

    Episode metadata
    ----------------
    step                 : current step within episode
    difficulty           : "easy" | "medium" | "hard"
    scenario_id          : name of active scenario (None if free-play)
    scenario_description : human-readable scenario description

    Step result fields (None on reset)
    --------------------
    reward               : scalar reward [-1, +1]
    reward_components    : per-component reward breakdown
    done                 : True when episode ends
    relay_tripped        : True if a relay trip happened this step
    disturbance          : type and description of any disturbance this step
    """

    # Physics
    bus_angles_deg:    List[float]
    line_flows_mw:     List[float]
    line_loading_frac: List[float]
    line_status:       List[bool]
    gen_dispatch_mw:   List[float]
    gen_available_mw:  List[float]
    gen_online:        List[bool]
    bus_loads_mw:      List[float]
    total_load_mw:     float
    total_gen_mw:      float
    power_balance_mw:  float
    cf_wind:           float
    cf_solar:          float

    # Episode metadata
    step:               int
    difficulty:         str
    scenario_id:        Optional[str] = None
    scenario_description: Optional[str] = None

    # Step result (None on reset)
    reward:             Optional[float] = None
    reward_components:  Optional[Dict[str, float]] = None
    done:               Optional[bool] = None
    relay_tripped:      Optional[bool] = None
    disturbance:        Optional[Dict] = None


# ─────────────────────────────────────────────────────────────────────────────
# State  (server-side persistent state)
# ─────────────────────────────────────────────────────────────────────────────

class PowerGridState(BaseModel):
    """Server-side state snapshot (returned by GET /state).

    Extends observation with grading-related episode statistics.
    """
    observation:        PowerGridObservation
    episode_reward:     float = 0.0
    overload_events:    int   = 0
    relay_trips:        int   = 0
    gen_outages:        int   = 0
    steps_elapsed:      int   = 0


# ─────────────────────────────────────────────────────────────────────────────
# Reset / Step request models (used internally by server)
# ─────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty:  str            = "easy"
    scenario_id: Optional[str] = None
    seed:        Optional[int] = None


class StepRequest(BaseModel):
    action: List[float]
