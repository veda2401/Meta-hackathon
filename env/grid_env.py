"""
env/grid_env.py
---------------
Power Grid Optimization Environment — IEEE 14-Bus, DC Power Flow.

Key technical features
----------------------
• Real DC power flow (B-matrix linear solver via numpy) — not a toy simulation
• IEEE 14-bus topology: 14 buses, 20 transmission lines, 6 generators / 5 fuels
• Dense shaped reward with 6 components, range [-1, +1]
• Stochastic load noise + 3 disturbance event types on MEDIUM / HARD
• Relay protection logic: blocks overloaded lines (threshold × rating)
• Full OpenEnv spec: step(), reset(), state()
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from env import ieee14
from env.dc_power_flow import DCResult, solve_dc_power_flow


# ── Difficulty ────────────────────────────────────────────────────────────────

class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ── Disturbance event types ───────────────────────────────────────────────────

class DisturbanceType(str, Enum):
    LINE_TRIP  = "line_trip"    # random line fault
    LOAD_SPIKE = "load_spike"   # sudden load surge at a bus
    GEN_OUTAGE = "gen_outage"   # generator forced offline


# ── Per-difficulty configuration ──────────────────────────────────────────────

@dataclass(frozen=True)
class DifficultyProfile:
    # Stochastic load
    load_noise_std_frac: float     # σ as fraction of base load per bus
    load_spike_mag_frac: float     # extra load fraction during LOAD_SPIKE

    # Wind / solar capacity factor: Gaussian random walk bounds
    cf_noise_std: float            # per-step σ for capacity-factor random walk
    cf_min: float                  # floor for renewable capacity factor
    cf_max: float                  # ceiling

    # Disturbances
    disturbance_prob: float        # probability of any disturbance per step
    disturbance_types: Tuple       # allowed disturbance types

    # Relay protection
    relay_threshold: float         # trip when |flow| > threshold × rating
    line_restore_steps: int        # steps before a relay-tripped line auto-restores

    # Reward weights (must sum to 1.0)
    reward_weights: Dict[str, float]

    # Episode length
    max_steps: int

    # Ramp-rate multiplier applied on top of GEN_RAMP baseline
    ramp_multiplier: float


PROFILES: Dict[Difficulty, DifficultyProfile] = {
    Difficulty.EASY: DifficultyProfile(
        load_noise_std_frac=0.02,
        load_spike_mag_frac=0.00,
        cf_noise_std=0.01,
        cf_min=0.70,
        cf_max=1.00,
        disturbance_prob=0.00,
        disturbance_types=(),
        relay_threshold=1.00,        # only trips at exact limit
        line_restore_steps=5,
        reward_weights=dict(balance=0.40, overload=0.20, reserve=0.15,
                             renewable=0.10, cost=0.10, stability=0.05),
        max_steps=100,
        ramp_multiplier=1.5,
    ),
    Difficulty.MEDIUM: DifficultyProfile(
        load_noise_std_frac=0.08,
        load_spike_mag_frac=0.25,
        cf_noise_std=0.05,
        cf_min=0.40,
        cf_max=1.00,
        disturbance_prob=0.04,
        disturbance_types=(DisturbanceType.LOAD_SPIKE, DisturbanceType.GEN_OUTAGE),
        relay_threshold=0.95,
        line_restore_steps=10,
        reward_weights=dict(balance=0.30, overload=0.25, reserve=0.15,
                             renewable=0.10, cost=0.10, stability=0.10),
        max_steps=150,
        ramp_multiplier=1.0,
    ),
    Difficulty.HARD: DifficultyProfile(
        load_noise_std_frac=0.15,
        load_spike_mag_frac=0.45,
        cf_noise_std=0.10,
        cf_min=0.10,
        cf_max=1.00,
        disturbance_prob=0.08,
        disturbance_types=(
            DisturbanceType.LINE_TRIP,
            DisturbanceType.LOAD_SPIKE,
            DisturbanceType.GEN_OUTAGE,
        ),
        relay_threshold=0.90,
        line_restore_steps=20,
        reward_weights=dict(balance=0.25, overload=0.25, reserve=0.15,
                             renewable=0.10, cost=0.10, stability=0.15),
        max_steps=200,
        ramp_multiplier=0.7,
    ),
}


# ── Main environment ──────────────────────────────────────────────────────────

class PowerGridEnv:
    """
    IEEE 14-Bus Power Grid Optimization Environment.

    Action space
    ------------
    ndarray(6,) — desired MW output for each generator.
    action[i] must be in [GEN_P_MIN[i], GEN_P_MAX[i]].
    Wind / solar are further bounded by their current capacity factor.

    State (observation) space
    -------------------------
    dict — see state() docstring for full key listing.

    Reward
    ------
    Scalar in [-1, +1] — weighted average of 6 physics-based components.
    """

    def __init__(
        self,
        difficulty: Difficulty | str = Difficulty.EASY,
        seed: Optional[int] = None,
    ):
        if isinstance(difficulty, str):
            difficulty = Difficulty(difficulty.lower())
        self.difficulty: Difficulty = difficulty
        self.profile: DifficultyProfile = PROFILES[difficulty]
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Mutable state (initialised in reset)
        self._step:          int         = 0
        self._done:          bool        = False
        self._ep_reward:     float       = 0.0
        self._gen_dispatch:  np.ndarray  = np.zeros(ieee14.N_GENS)
        self._gen_online:    List[bool]  = [True] * ieee14.N_GENS
        self._gen_outage_remaining: List[int] = [0] * ieee14.N_GENS
        self._bus_loads:     np.ndarray  = np.zeros(ieee14.N_BUSES)
        self._line_status:   List[bool]  = [True] * ieee14.N_LINES
        self._relay_restore: List[int]   = [0] * ieee14.N_LINES  # step when restorable
        self._cf_wind:       float       = 1.0
        self._cf_solar:      float       = 1.0
        self._last_dc:       Optional[DCResult] = None
        self._history:       List[dict]  = []

        self.reset()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        """Reset environment; return initial state observation."""
        p = self.profile
        self._step         = 0
        self._done         = False
        self._ep_reward    = 0.0
        self._history      = []
        self._gen_online   = [True] * ieee14.N_GENS
        self._gen_outage_remaining = [0] * ieee14.N_GENS
        self._line_status  = [True] * ieee14.N_LINES
        self._relay_restore = [0] * ieee14.N_LINES
        self._cf_wind      = self._rng.uniform(0.75, 1.00)
        self._cf_solar     = self._rng.uniform(0.70, 1.00)

        # Initial dispatch: proportional to capacity (feasible starting point)
        total_base = ieee14.TOTAL_BASE_LOAD_MW
        total_cap  = sum(ieee14.GEN_P_MAX)
        self._gen_dispatch = np.array([
            min(ieee14.GEN_P_MAX[i],
                max(ieee14.GEN_P_MIN[i],
                    ieee14.GEN_P_MAX[i] * (total_base / total_cap)))
            for i in range(ieee14.N_GENS)
        ])
        self._bus_loads = np.array(ieee14.BASE_LOAD_MW, dtype=float)

        self._last_dc = self._run_power_flow()
        return self.state()

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        """
        Advance the environment by one time step.

        Parameters
        ----------
        action : array-like, shape (6,)
            Desired MW setpoints for generators 0-5.

        Returns
        -------
        state   : dict
        reward  : float  — in [-1, +1]
        done    : bool
        info    : dict
        """
        if self._done:
            raise RuntimeError("Episode finished. Call reset().")

        action = self._validate_action(action)

        # 1. Apply ramp-rate and capacity limits
        new_dispatch = self._apply_ramp_limits(action)

        # 2. Stochastic load update
        self._update_loads()

        # 3. Update renewable capacity factors (random walk)
        self._update_capacity_factors()

        # 4. Clamp renewables by current capacity factor
        new_dispatch[3] = min(new_dispatch[3],
                              ieee14.GEN_P_MAX[3] * self._cf_wind)   # wind
        new_dispatch[5] = min(new_dispatch[5],
                              ieee14.GEN_P_MAX[5] * self._cf_solar)  # solar

        # 5. Zero out offline generators
        for i in range(ieee14.N_GENS):
            if not self._gen_online[i]:
                new_dispatch[i] = 0.0

        self._gen_dispatch = new_dispatch

        # 6. Check auto-restore of relayed lines
        self._auto_restore_lines()

        # 7. Apply random disturbances (MEDIUM / HARD)
        disturbance_info = self._apply_disturbances()

        # 8. DC power flow
        self._last_dc = self._run_power_flow()

        # 9. Relay protection (trip lines over threshold)
        relay_events = self._apply_relay_protection()

        # 10. Reward
        reward, reward_components = self._compute_reward(self._last_dc)
        self._ep_reward += reward

        self._step += 1
        done = self._step >= self.profile.max_steps
        self._done = done

        info = {
            "step":               self._step,
            "reward_components":  reward_components,
            "disturbance":        disturbance_info,
            "relay_tripped":      relay_events,
            "converged":          self._last_dc.converged,
            "total_load_mw":      float(self._bus_loads.sum()),
            "total_gen_mw":       float(self._gen_dispatch.sum()),
            "overloaded_lines":   self._last_dc.overloaded_lines,
            "episode_reward":     self._ep_reward,
        }
        self._history.append(info)
        return self.state(), float(reward), done, info

    def state(self) -> dict:
        """
        Return the current environment observation.

        Keys
        ----
        bus_angles_deg       : list[float]  — 14 voltage angles (deg, slack=0)
        line_flows_mw        : list[float]  — 20 line flows (signed MW)
        line_loading_frac    : list[float]  — |flow| / rating per line
        line_status          : list[bool]   — True = online
        gen_dispatch_mw      : list[float]  — 6 generator outputs
        gen_available_mw     : list[float]  — 6 max available (weather-limited)
        gen_online           : list[bool]   — 6 online flags
        bus_loads_mw         : list[float]  — 14 bus loads
        total_load_mw        : float
        total_gen_mw         : float
        power_balance_mw     : float        — gen − load
        cf_wind              : float        — wind capacity factor [0,1]
        cf_solar             : float        — solar capacity factor [0,1]
        step                 : int
        difficulty           : str
        """
        dc = self._last_dc
        if dc is None:
            dc = self._run_power_flow()

        gen_avail = list(ieee14.GEN_P_MAX)
        gen_avail[3] = ieee14.GEN_P_MAX[3] * self._cf_wind
        gen_avail[5] = ieee14.GEN_P_MAX[5] * self._cf_solar

        lf = dc.loading_fractions.tolist()

        return {
            "bus_angles_deg":    dc.angles_deg.tolist(),
            "line_flows_mw":     dc.line_flows_mw.tolist(),
            "line_loading_frac": lf,
            "line_status":       list(self._line_status),
            "gen_dispatch_mw":   self._gen_dispatch.tolist(),
            "gen_available_mw":  gen_avail,
            "gen_online":        list(self._gen_online),
            "bus_loads_mw":      self._bus_loads.tolist(),
            "total_load_mw":     float(self._bus_loads.sum()),
            "total_gen_mw":      float(self._gen_dispatch.sum()),
            "power_balance_mw":  float(self._gen_dispatch.sum() - self._bus_loads.sum()),
            "cf_wind":           self._cf_wind,
            "cf_solar":          self._cf_solar,
            "step":              self._step,
            "difficulty":        self.difficulty.value,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _validate_action(self, action) -> np.ndarray:
        a = np.asarray(action, dtype=float)
        if a.shape != (ieee14.N_GENS,):
            raise ValueError(
                f"action must have shape ({ieee14.N_GENS},), got {a.shape}"
            )
        return np.clip(a, ieee14.GEN_P_MIN, ieee14.GEN_P_MAX)

    def _apply_ramp_limits(self, target: np.ndarray) -> np.ndarray:
        p = self.profile
        result = np.array(self._gen_dispatch, dtype=float)
        for i in range(ieee14.N_GENS):
            ramp = ieee14.GEN_RAMP[i] * p.ramp_multiplier
            delta = target[i] - self._gen_dispatch[i]
            delta = np.clip(delta, -ramp, ramp)
            result[i] = np.clip(
                self._gen_dispatch[i] + delta,
                ieee14.GEN_P_MIN[i], ieee14.GEN_P_MAX[i]
            )
        return result

    def _update_loads(self) -> None:
        """Gaussian noise on base loads, scaled by difficulty profile."""
        p = self.profile
        for i in range(ieee14.N_BUSES):
            base = ieee14.BASE_LOAD_MW[i]
            if base == 0.0:
                self._bus_loads[i] = 0.0
                continue
            noise = self._np_rng.normal(0, base * p.load_noise_std_frac)
            self._bus_loads[i] = max(0.0, base + noise)

    def _update_capacity_factors(self) -> None:
        p = self.profile
        self._cf_wind  = float(np.clip(
            self._cf_wind  + self._np_rng.normal(0, p.cf_noise_std),
            p.cf_min, p.cf_max
        ))
        self._cf_solar = float(np.clip(
            self._cf_solar + self._np_rng.normal(0, p.cf_noise_std),
            p.cf_min, p.cf_max
        ))

    def _auto_restore_lines(self) -> None:
        """Restore relay-tripped lines after the restoration delay expires."""
        for i in range(ieee14.N_LINES):
            if not self._line_status[i] and self._relay_restore[i] <= self._step:
                self._line_status[i] = True

    def _apply_disturbances(self) -> dict:
        """Fire one disturbance event per step with profile probability."""
        p = self.profile
        if not p.disturbance_types or self._rng.random() > p.disturbance_prob:
            return {"type": None}

        dtype = self._rng.choice(p.disturbance_types)
        info: dict = {"type": dtype.value}

        if dtype == DisturbanceType.LINE_TRIP:
            online = [i for i, s in enumerate(self._line_status) if s]
            if online:
                tripped = self._rng.choice(online)
                self._line_status[tripped] = False
                self._relay_restore[tripped] = (
                    self._step + self.profile.line_restore_steps
                )
                info["line_idx"] = tripped

        elif dtype == DisturbanceType.LOAD_SPIKE:
            load_buses = [i for i in range(ieee14.N_BUSES) if ieee14.BASE_LOAD_MW[i] > 0]
            bus = self._rng.choice(load_buses)
            extra = ieee14.BASE_LOAD_MW[bus] * p.load_spike_mag_frac
            self._bus_loads[bus] += extra
            info["bus_idx"] = bus
            info["extra_mw"] = extra

        elif dtype == DisturbanceType.GEN_OUTAGE:
            online_gens = [i for i, o in enumerate(self._gen_online) if o]
            if online_gens:
                gen = self._rng.choice(online_gens)
                self._gen_online[gen] = False
                self._gen_dispatch[gen] = 0.0
                # Restore after 10-20 steps
                self._gen_outage_remaining[gen] = self._rng.randint(10, 20)
                info["gen_idx"] = gen

        # Tick down gen outage counters
        for i in range(ieee14.N_GENS):
            if not self._gen_online[i]:
                self._gen_outage_remaining[i] -= 1
                if self._gen_outage_remaining[i] <= 0:
                    self._gen_online[i] = True

        return info

    def _apply_relay_protection(self) -> List[int]:
        """
        Trip lines where |flow| ≥ relay_threshold × rating.
        Relay protection blocks re-closing until line_restore_steps elapse.
        """
        p = self.profile
        tripped: List[int] = []
        for i in range(ieee14.N_LINES):
            if not self._line_status[i]:
                continue
            flow   = abs(self._last_dc.line_flows_mw[i])
            rating = ieee14.LINE_RATINGS_MW[i]
            if flow >= p.relay_threshold * rating:
                self._line_status[i] = False
                self._relay_restore[i] = self._step + p.line_restore_steps
                tripped.append(i)
        return tripped

    def _run_power_flow(self) -> DCResult:
        """
        Build net injection vector and run the DC power flow solver.
        """
        p_inj = np.zeros(ieee14.N_BUSES, dtype=float)

        # Inject from generators
        for i in range(ieee14.N_GENS):
            if self._gen_online[i]:
                p_inj[ieee14.GEN_BUS_IDX[i]] += self._gen_dispatch[i]

        # Subtract loads
        p_inj -= self._bus_loads

        return solve_dc_power_flow(p_inj, self._line_status)

    def _compute_reward(self, dc: DCResult) -> Tuple[float, dict]:
        """
        Dense shaped reward — 6 components, each in [-1, +1].
        Final reward = weighted average (guaranteed in [-1, +1]).
        """
        p          = self.profile
        total_load = float(self._bus_loads.sum())
        total_gen  = float(self._gen_dispatch.sum())

        # ── 1. Balance ────────────────────────────────────────────────────────
        # ±0 MW error → +1.0;  ≥ 50 % of load imbalance → -1.0
        imb_frac   = abs(total_gen - total_load) / max(total_load, 1.0)
        r_balance  = max(-1.0, 1.0 - 2.0 * min(1.0, imb_frac / 0.50))

        # ── 2. Overload ───────────────────────────────────────────────────────
        # sum of per-line excess fractions, clipped to [0, 1] → [-1, 0]
        excess = sum(
            max(0.0, abs(dc.line_flows_mw[i]) / max(ieee14.LINE_RATINGS_MW[i], 1.0) - 1.0)
            for i in range(ieee14.N_LINES)
            if self._line_status[i]
        )
        r_overload = max(-1.0, -min(1.0, excess))

        # ── 3. Spinning reserve ───────────────────────────────────────────────
        # need ≥ 10 % of load as spare online capacity
        online_capacity = sum(
            ieee14.GEN_P_MAX[i] * self._cf_wind if ieee14.GEN_FUEL[i] == "wind"
            else ieee14.GEN_P_MAX[i] * self._cf_solar if ieee14.GEN_FUEL[i] == "solar"
            else ieee14.GEN_P_MAX[i]
            for i in range(ieee14.N_GENS) if self._gen_online[i]
        )
        reserve_needed = 0.10 * total_load
        reserve_actual = max(0.0, online_capacity - total_gen)
        r_reserve = max(-1.0, min(1.0,
            2.0 * min(1.0, reserve_actual / max(reserve_needed, 1.0)) - 1.0
        ))

        # ── 4. Renewable utilisation ──────────────────────────────────────────
        # target: serve 30 % of load from wind + solar
        ren_gen    = sum(self._gen_dispatch[i] for i in ieee14.RENEWABLE_IDX)
        ren_target = 0.30 * total_load
        r_renewable = max(-1.0, min(1.0,
            2.0 * min(1.0, ren_gen / max(ren_target, 1.0)) - 1.0
        ))

        # ── 5. Fuel cost ──────────────────────────────────────────────────────
        cost = sum(
            ieee14.GEN_COST_A[i] * self._gen_dispatch[i] ** 2
            + ieee14.GEN_COST_B[i] * self._gen_dispatch[i]
            for i in range(ieee14.N_GENS)
        )
        # Normalise by max possible cost; low cost → high reward
        r_cost = max(-1.0, min(1.0, 1.0 - 2.0 * cost / max(ieee14.MAX_COST, 1.0)))

        # ── 6. Voltage angle stability ────────────────────────────────────────
        # max angle difference across any line; 30° limit → -1
        r_stability = max(-1.0, min(1.0,
            1.0 - 2.0 * min(1.0, dc.max_angle_diff_deg / 30.0)
        ))

        # ── Weighted total ────────────────────────────────────────────────────
        w = p.reward_weights
        total_reward = (
            w["balance"]   * r_balance  +
            w["overload"]  * r_overload +
            w["reserve"]   * r_reserve  +
            w["renewable"] * r_renewable +
            w["cost"]      * r_cost     +
            w["stability"] * r_stability
        )

        components = {
            "balance":   round(r_balance,   4),
            "overload":  round(r_overload,  4),
            "reserve":   round(r_reserve,   4),
            "renewable": round(r_renewable, 4),
            "cost":      round(r_cost,      4),
            "stability": round(r_stability, 4),
            "total":     round(total_reward, 4),
        }
        return total_reward, components

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_generators(self) -> int:
        return ieee14.N_GENS

    @property
    def n_buses(self) -> int:
        return ieee14.N_BUSES

    @property
    def n_lines(self) -> int:
        return ieee14.N_LINES

    @property
    def max_steps(self) -> int:
        return self.profile.max_steps

    @property
    def episode_reward(self) -> float:
        return self._ep_reward

    @property
    def history(self) -> List[dict]:
        return list(self._history)

    def __repr__(self) -> str:
        return (
            f"PowerGridEnv(difficulty={self.difficulty.value}, "
            f"step={self._step}/{self.max_steps}, "
            f"ep_reward={self._ep_reward:.3f})"
        )
