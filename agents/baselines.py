"""
agents/baselines.py
-------------------
Baseline agents for the IEEE 14-Bus Power Grid Environment.

RandomAgent           – random feasible dispatch (lower bound)
RuleBasedAgent        – proportional load-share, difficulty-aware buffer
EconomicDispatchAgent – merit-order (cheapest-first) greedy dispatch
"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np

from env.grid_env import PowerGridEnv, Difficulty
from env import ieee14


class BaseAgent:
    name: str = "BaseAgent"

    def act(self, state: dict, env: PowerGridEnv) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.name}()"


# ── 1. Random Agent ────────────────────────────────────────────────────────

class RandomAgent(BaseAgent):
    """Uniform random dispatch within [p_min, p_available] per generator."""

    name = "RandomAgent"

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def act(self, state: dict, env: PowerGridEnv) -> np.ndarray:
        avail  = state["gen_available_mw"]
        online = state["gen_online"]
        action = np.zeros(ieee14.N_GENS)
        for i in range(ieee14.N_GENS):
            if online[i]:
                lo = ieee14.GEN_P_MIN[i]
                hi = min(ieee14.GEN_P_MAX[i], avail[i])
                action[i] = float(self._rng.uniform(lo, hi))
        return action


# ── 2. Rule-Based Agent ────────────────────────────────────────────────────

class RuleBasedAgent(BaseAgent):
    """
    Proportional load-sharing with a difficulty-tuned safety buffer.

    Strategy
    --------
    1. Target total generation = observed_load × (1 + buffer).
    2. Share target equally among online generators, weighted by capacity.
    3. Clip to each generator's [p_min, p_available] window.
    """

    name = "RuleBasedAgent"

    _BUFFER = {Difficulty.EASY: 0.03, Difficulty.MEDIUM: 0.07, Difficulty.HARD: 0.12}

    def act(self, state: dict, env: PowerGridEnv) -> np.ndarray:
        buf     = self._BUFFER[env.difficulty]
        load    = state["total_load_mw"]
        target  = load * (1.0 + buf)
        avail   = state["gen_available_mw"]
        online  = state["gen_online"]

        total_cap = sum(avail[i] for i in range(ieee14.N_GENS) if online[i]) or 1.0

        action = np.zeros(ieee14.N_GENS)
        for i in range(ieee14.N_GENS):
            if not online[i]:
                continue
            share    = target * (avail[i] / total_cap)
            action[i] = float(np.clip(share, ieee14.GEN_P_MIN[i], avail[i]))
        return action


# ── 3. Economic Dispatch Agent ─────────────────────────────────────────────

class EconomicDispatchAgent(BaseAgent):
    """
    Merit-order economic dispatch.

    Generators are sorted by marginal cost (cost_b $/MWh, then cost_a ×P).
    Cheapest units are loaded to their maximum first; remaining units fill gap.
    Respects gen_available_mw (weather-limited renewables) and gen_online flags.

    This is the strongest non-learned baseline — it minimises fuel cost while
    targeting load balance.
    """

    name = "EconomicDispatchAgent"

    _RESERVE = {Difficulty.EASY: 0.02, Difficulty.MEDIUM: 0.07, Difficulty.HARD: 0.05}

    def act(self, state: dict, env: PowerGridEnv) -> np.ndarray:
        load    = state["total_load_mw"]
        avail   = state["gen_available_mw"]
        online  = state["gen_online"]
        reserve = self._RESERVE[env.difficulty]
        target  = load * (1.0 + reserve)

        # Sort generators by marginal cost at midpoint dispatch
        def marginal_cost(i: int) -> float:
            mid = (ieee14.GEN_P_MIN[i] + avail[i]) / 2.0
            return ieee14.GEN_COST_A[i] * 2 * mid + ieee14.GEN_COST_B[i]

        order = sorted(
            [i for i in range(ieee14.N_GENS) if online[i]],
            key=marginal_cost,
        )

        action    = np.zeros(ieee14.N_GENS)
        remaining = target

        # First pass: commit units to p_min
        for i in order:
            action[i] = ieee14.GEN_P_MIN[i]
            remaining -= action[i]

        # Second pass: fill from cheapest to most expensive
        for i in order:
            headroom = float(np.clip(avail[i], ieee14.GEN_P_MIN[i], avail[i])) - action[i]
            extra    = min(headroom, max(0.0, remaining))
            action[i] += extra
            remaining  -= extra
            if remaining <= 1e-3:
                break

        # HARD: avoid over-generation waste (trim proportionally)
        if env.difficulty == Difficulty.HARD:
            total = float(action.sum())
            if total > load * 1.06:
                scale = (load * 1.06) / total
                for i in range(ieee14.N_GENS):
                    action[i] = max(
                        ieee14.GEN_P_MIN[i] if online[i] else 0.0,
                        action[i] * scale
                    )

        return action
