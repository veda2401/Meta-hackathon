"""
tasks/graders.py
----------------
Episode evaluation for the IEEE 14-Bus Power Grid Environment.

Per-difficulty pass criteria and 0-100 scoring mapped to letter grades.
"""

from __future__ import annotations

from dataclasses import dataclass

from env.grid_env import Difficulty, PowerGridEnv


GRADE_THRESHOLDS = [(90, "A"), (75, "B"), (60, "C"), (40, "D")]


def _letter(score: float) -> str:
    for thr, g in GRADE_THRESHOLDS:
        if score >= thr:
            return g
    return "F"


@dataclass(frozen=True)
class PassCriteria:
    min_avg_reward: float      # must achieve at least this avg reward/step
    max_overload_events: int   # cumulative line-overload events
    max_relay_trips: int       # cumulative relay-trip events
    max_gen_outages: int       # cumulative generator trips


CRITERIA: dict[Difficulty, PassCriteria] = {
    Difficulty.EASY:   PassCriteria(min_avg_reward=-0.20, max_overload_events=30,
                                    max_relay_trips=30, max_gen_outages=0),
    Difficulty.MEDIUM: PassCriteria(min_avg_reward=-0.45, max_overload_events=40,
                                    max_relay_trips=40, max_gen_outages=6),
    Difficulty.HARD:   PassCriteria(min_avg_reward=-0.70, max_overload_events=60,
                                    max_relay_trips=60, max_gen_outages=15),
}


def grade_episode(env: PowerGridEnv) -> dict:
    """
    Grade a completed (or partial) episode.

    Scoring (0–100)
    ---------------
    40 pts  reward performance (normalised across difficulty range)
    20 pts  overload events
    20 pts  relay trips
    10 pts  gen outages
    10 pts  DC power flow convergence rate
    """
    history = env.history
    if not history:
        raise ValueError("No episode history — run at least one step first.")

    difficulty = env.difficulty
    crit       = CRITERIA[difficulty]

    # ── Aggregate stats ────────────────────────────────────────────────────
    total_steps      = len(history)
    total_reward     = sum(h["reward_components"]["total"] for h in history)
    avg_reward       = total_reward / total_steps
    overload_events  = sum(len(h["overloaded_lines"]) for h in history)
    relay_trips      = sum(len(h["relay_tripped"]) for h in history)
    gen_outages      = sum(
        1 for h in history
        if h["disturbance"].get("type") == "gen_outage"
    )
    converged_steps  = sum(1 for h in history if h["converged"])

    # ── Component scores ───────────────────────────────────────────────────
    # Reward score: map avg_reward from [-1, +1] to [0, 40]
    reward_score = max(0.0, min(40.0, (avg_reward + 1.0) / 2.0 * 40.0))

    # Overload score: 20 pts, linear decay
    ol_ratio      = overload_events / max(1, crit.max_overload_events)
    overload_score = max(0.0, 20.0 * (1.0 - min(1.0, ol_ratio)))

    # Relay score: 20 pts
    rel_ratio   = relay_trips / max(1, crit.max_relay_trips)
    relay_score = max(0.0, 20.0 * (1.0 - min(1.0, rel_ratio)))

    # Outage score: 10 pts
    out_ratio    = gen_outages / max(1, crit.max_gen_outages)
    outage_score = max(0.0, 10.0 * (1.0 - min(1.0, out_ratio)))

    # Convergence score: 10 pts
    conv_score = 10.0 * converged_steps / total_steps

    total_score = reward_score + overload_score + relay_score + outage_score + conv_score
    
    # STRICT OpenEnv Validator Fix: The validator parses all task scores and assert 0 < score < 1. 
    # We must ensure that total_score is scaled to [0.01, 0.99].
    total_score = max(1.0, min(99.0, total_score))
    clamped_score_01 = round(total_score / 100.0, 4)

    passed = (
        avg_reward       >= crit.min_avg_reward and
        overload_events  <= crit.max_overload_events and
        relay_trips      <= crit.max_relay_trips and
        gen_outages      <= crit.max_gen_outages
    )

    # Average reward components across episode
    avg_comps: dict[str, float] = {}
    for key in ("balance", "overload", "reserve", "renewable", "cost", "stability"):
        avg_comps[key] = round(
            sum(h["reward_components"][key] for h in history) / total_steps, 4
        )

    # We must ensure that total_score is scaled to strictly [1.0, 99.0].
    # This prevents edge case outputs of precisely 0.0 or 100.0, which 
    # OpenEnv normalizes down to EXACTLY 0.0 or 1.0 (crashing Deep Validator checks).
    total_score = max(1.0, min(99.0, total_score))

    return {
        "difficulty":   difficulty.value,
        "total_score":  round(total_score, 2),
        "score_01":     round(total_score / 100.0, 4),
        "score":        round(total_score / 100.0, 4),
        "grade":        _letter(total_score),
        "passed":       passed,
        "breakdown": {
            "reward_pts":   round(reward_score,   2),
            "overload_pts": round(overload_score, 2),
            "relay_pts":    round(relay_score,    2),
            "outage_pts":   round(outage_score,   2),
            "conv_pts":     round(conv_score,     2),
        },
        "metrics": {
            "total_steps":        total_steps,
            "total_reward":       round(total_reward, 4),
            "avg_reward/step":    round(avg_reward,   4),
            "overload_events":    overload_events,
            "relay_trips":        relay_trips,
            "gen_outages":        gen_outages,
            "convergence_rate":   round(converged_steps / max(1, total_steps), 4),
        },
        "avg_reward_components": avg_comps,
    }
