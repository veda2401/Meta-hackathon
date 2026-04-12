"""
tasks/graders.py
----------------
Episode evaluation for the IEEE 14-Bus Power Grid Environment.

Per-difficulty pass criteria. All score fields are STRICTLY in (0, 1) —
boundary values 0.0 and 1.0 are structurally impossible by design.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from env.grid_env import Difficulty, PowerGridEnv


GRADE_THRESHOLDS = [(90, "A"), (75, "B"), (60, "C"), (40, "D")]


def _letter(pts: float) -> str:
    for thr, g in GRADE_THRESHOLDS:
        if pts >= thr:
            return g
    return "F"


def _safe(v: float) -> float:
    """
    Map any float to STRICTLY (0.05, 0.95).

    Uses a sigmoid-inspired transform so:
      - input 0.0  → 0.5 (sigmoid center)
      - input 1.0  → ~0.731
      - input -5   → ~0.07
      - input  5   → ~0.927
    Then linearly rescale sigmoid output into (0.05, 0.95).

    No matter what floats come in, the output is always well inside (0, 1).
    """
    sig = 1.0 / (1.0 + math.exp(-v))        # sigmoid: (0, 1) exclusive always
    result = 0.05 + 0.90 * sig              # rescale to (0.05, 0.95)
    return round(float(result), 4)


def _clamp01(v: float) -> float:
    """Clamp raw [0,1] normalized value to strict (0.05, 0.95) using linear map."""
    # v is expected in [0.0, 1.0]; map to [0.05, 0.95]
    clamped = 0.05 + 0.90 * max(0.0, min(1.0, float(v)))
    return round(float(clamped), 4)


@dataclass(frozen=True)
class PassCriteria:
    min_avg_reward: float
    max_overload_events: int
    max_relay_trips: int
    max_gen_outages: int


CRITERIA: dict[Difficulty, PassCriteria] = {
    Difficulty.EASY: PassCriteria(
        min_avg_reward=-0.20,
        max_overload_events=30,
        max_relay_trips=30,
        max_gen_outages=0,
    ),
    Difficulty.MEDIUM: PassCriteria(
        min_avg_reward=-0.45,
        max_overload_events=40,
        max_relay_trips=40,
        max_gen_outages=6,
    ),
    Difficulty.HARD: PassCriteria(
        min_avg_reward=-0.70,
        max_overload_events=60,
        max_relay_trips=60,
        max_gen_outages=15,
    ),
}


def grade_episode(env: PowerGridEnv) -> dict:
    """
    Grade a completed episode.

    Returns a dict where 'score' and 'score_01' are STRICTLY in (0.05, 0.95).
    No field in the returned dict can ever be exactly 0.0 or 1.0.
    """
    history = env.history
    if not history:
        raise ValueError("No episode history — run at least one step first.")

    difficulty = env.difficulty
    crit = CRITERIA[difficulty]

    # ── Raw aggregates ───────────────────────────────────────────────────────
    total_steps     = len(history)
    total_reward    = sum(h["reward_components"]["total"] for h in history)
    avg_reward      = total_reward / total_steps
    overload_events = sum(len(h["overloaded_lines"]) for h in history)
    relay_trips     = sum(len(h["relay_tripped"]) for h in history)
    gen_outages     = sum(
        1 for h in history if h["disturbance"].get("type") == "gen_outage"
    )
    converged_steps = sum(1 for h in history if h["converged"])

    # ── Component ratios (all in [0, 1]) ─────────────────────────────────────
    # Reward: avg_reward in [-1, +1] → normalized to [0, 1]
    reward_ratio   = (avg_reward + 1.0) / 2.0

    # Event ratios: 0 events = best (1.0), at/above cap = worst (0.0)
    overload_ratio = 1.0 - min(1.0, overload_events / max(1, crit.max_overload_events))
    relay_ratio    = 1.0 - min(1.0, relay_trips     / max(1, crit.max_relay_trips))
    outage_ratio   = 1.0 - min(1.0, gen_outages     / max(1, crit.max_gen_outages))
    conv_ratio     = converged_steps / total_steps

    # ── Weighted composite in [0, 1] ─────────────────────────────────────────
    raw_ratio = (
        0.40 * reward_ratio   +   # 40% weight
        0.20 * overload_ratio +   # 20% weight
        0.20 * relay_ratio    +   # 20% weight
        0.10 * outage_ratio   +   # 10% weight
        0.10 * conv_ratio         # 10% weight
    )
    # raw_ratio is guaranteed in [0, 1] since all components are in [0, 1]

    # ── Map to STRICT (0.05, 0.95) — structurally impossible to hit 0 or 1 ──
    safe_score = _clamp01(raw_ratio)

    # ── 0-100 display scale ───────────────────────────────────────────────────
    display_pts = round(float(raw_ratio) * 100.0, 2)
    # Keep display in a safe display range (never output an exact integer 0/100)
    display_pts = round(max(0.01, min(99.99, display_pts)), 2)

    # ── Pass/fail ─────────────────────────────────────────────────────────────
    passed = (
        avg_reward      >= crit.min_avg_reward
        and overload_events <= crit.max_overload_events
        and relay_trips     <= crit.max_relay_trips
        and gen_outages     <= crit.max_gen_outages
    )

    # ── Safe ratio fields (never 0.0 or 1.0) ─────────────────────────────────
    safe_conv_rate = _clamp01(conv_ratio)

    return {
        "difficulty":   difficulty.value,
        "total_points": display_pts,
        "score_01":     safe_score,
        "score":        safe_score,
        "grade":        _letter(display_pts),
        "passed":       passed,
        "metrics": {
            "total_steps":      total_steps,
            "avg_reward_step":  round(float(avg_reward), 4),
            "overload_events":  overload_events,
            "relay_trips":      relay_trips,
            "gen_outages":      gen_outages,
            "convergence_rate": safe_conv_rate,
        },
    }
