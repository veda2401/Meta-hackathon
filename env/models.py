"""
env/models.py
-------------
Pydantic typed models for the OpenEnv spec compliance.

These are the canonical types for:
  - ObservationModel  — what step() / state() return
  - ActionModel       — what step() accepts
  - RewardModel       — reward breakdown returned inside step() info
  - StepResult        — full (obs, reward, done, info) tuple as a model
"""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ObservationModel(BaseModel):
    """Typed observation returned by reset() and state()."""

    bus_angles_deg:    List[float] = Field(..., description="Voltage angles per bus (deg)")
    line_flows_mw:     List[float] = Field(..., description="Signed DC power flows (MW)")
    line_loading_frac: List[float] = Field(..., description="|flow|/rating per line [0,∞)")
    line_status:       List[bool]  = Field(..., description="True=online, False=relay-tripped")
    gen_dispatch_mw:   List[float] = Field(..., description="Generator outputs (MW)")
    gen_available_mw:  List[float] = Field(..., description="Max available per gen (MW, weather-limited)")
    gen_online:        List[bool]  = Field(..., description="Generator online flags")
    bus_loads_mw:      List[float] = Field(..., description="Stochastic bus loads (MW)")
    total_load_mw:     float       = Field(..., description="Σ bus loads (MW)")
    total_gen_mw:      float       = Field(..., description="Σ generator dispatch (MW)")
    power_balance_mw:  float       = Field(..., description="gen − load (MW)")
    cf_wind:           float       = Field(..., ge=0.0, le=1.0, description="Wind capacity factor")
    cf_solar:          float       = Field(..., ge=0.0, le=1.0, description="Solar capacity factor")
    step:              int         = Field(..., ge=0)
    difficulty:        str         = Field(..., pattern="^(easy|medium|hard)$")

    @field_validator("bus_angles_deg")
    @classmethod
    def _check_14_buses(cls, v: List[float]) -> List[float]:
        if len(v) != 14:
            raise ValueError(f"Expected 14 bus angles, got {len(v)}")
        return v

    @field_validator("line_flows_mw")
    @classmethod
    def _check_20_lines(cls, v: List[float]) -> List[float]:
        if len(v) != 20:
            raise ValueError(f"Expected 20 line flows, got {len(v)}")
        return v

    @field_validator("gen_dispatch_mw")
    @classmethod
    def _check_6_gens(cls, v: List[float]) -> List[float]:
        if len(v) != 6:
            raise ValueError(f"Expected 6 generator values, got {len(v)}")
        return v

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "bus_angles_deg": [0.0] * 14,
                "line_flows_mw": [0.0] * 20,
                "line_loading_frac": [0.0] * 20,
                "line_status": [True] * 20,
                "gen_dispatch_mw": [80.0, 60.0, 40.0, 35.0, 30.0, 20.0],
                "gen_available_mw": [100.0, 80.0, 50.0, 50.0, 40.0, 30.0],
                "gen_online": [True] * 6,
                "bus_loads_mw": [0.0, 21.7, 94.2, 47.8, 7.6, 11.2, 0.0, 0.0,
                                  29.5, 9.0, 3.5, 6.1, 13.5, 14.9],
                "total_load_mw": 259.0,
                "total_gen_mw": 265.0,
                "power_balance_mw": 6.0,
                "cf_wind": 0.85,
                "cf_solar": 0.70,
                "step": 0,
                "difficulty": "easy",
            }
        }
    )


class ActionModel(BaseModel):
    """Typed action passed to step()."""

    dispatch_mw: List[float] = Field(
        ...,
        min_length=6,
        max_length=6,
        description=(
            "Desired MW output for generators 0–5 "
            "[coal, gas, gas, wind, hydro, solar]. "
            "Values clipped to [p_min, min(p_max, p_available)]."
        ),
    )

    @field_validator("dispatch_mw")
    @classmethod
    def _non_negative(cls, v: List[float]) -> List[float]:
        if any(x < 0 for x in v):
            raise ValueError("All dispatch values must be ≥ 0")
        return v

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {"dispatch_mw": [80.0, 60.0, 40.0, 45.0, 35.0, 25.0]}
        }
    )


class RewardModel(BaseModel):
    """Typed reward breakdown returned in step() info dict."""

    balance:   float = Field(..., ge=-1.0, le=1.0,
                              description="Generation–load balance component")
    overload:  float = Field(..., ge=-1.0, le=0.0,
                              description="Line overload penalty")
    reserve:   float = Field(..., ge=-1.0, le=1.0,
                              description="Spinning reserve adequacy")
    renewable: float = Field(..., ge=-1.0, le=1.0,
                              description="Renewable utilisation vs 30% target")
    cost:      float = Field(..., ge=-1.0, le=1.0,
                              description="Normalised fuel cost (lower = better)")
    stability: float = Field(..., ge=-1.0, le=1.0,
                              description="Voltage angle stability")
    total:     float = Field(..., ge=-1.0, le=1.0,
                              description="Weighted total reward")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "balance": 0.82, "overload": 0.0, "reserve": 0.45,
                "renewable": 0.30, "cost": 0.55, "stability": 0.90,
                "total": 0.58,
            }
        }
    )


class StepResult(BaseModel):
    """Full typed result of one environment step."""

    observation: ObservationModel
    reward:      float = Field(..., ge=-1.0, le=1.0)
    done:        bool
    reward_components: RewardModel
    step:        int = Field(..., ge=1)


# ── Convenience helpers ────────────────────────────────────────────────────

def observation_from_dict(d: dict) -> ObservationModel:
    """Wrap a raw state dict in a typed ObservationModel."""
    return ObservationModel(**d)


def reward_from_dict(d: dict) -> RewardModel:
    """Wrap a raw reward_components dict in a typed RewardModel."""
    return RewardModel(**d)
