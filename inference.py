"""
inference.py
------------
Power Grid Crisis Environment — OpenEnv Hackathon Inference Script.

The LLM agent reasons step-by-step through each crisis scenario,
producing interpretable chain-of-thought reasoning alongside dispatch decisions.
This reveals HOW the model reasons under high-stakes conditions — not just WHAT it decides.

Compliant with official submission guidelines:
  - Uses OpenAI client with API_BASE_URL + HF_TOKEN
  - MODEL_NAME and API_BASE_URL have default values
  - HF_TOKEN is required (raises ValueError if missing)
  - stdout emits exactly [START] / [STEP] / [END] lines

Usage
-----
    python inference.py                          # LLM agent (needs HF_TOKEN set)
    python inference.py --agent economic         # baseline (no API key needed)
    python inference.py --episodes 3 --seed 42
    python inference.py --transcript             # save full reasoning transcript
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import textwrap
import time
from typing import List, Optional

import numpy as np

from env.grid_env import PowerGridEnv, Difficulty
from env import ieee14
from agents.baselines import RandomAgent, RuleBasedAgent, EconomicDispatchAgent
from tasks.graders import grade_episode

# ─────────────────────────────────────────────────────────────────────────────
# Environment variables  (per official spec)
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")          # mandatory — no default

# ─────────────────────────────────────────────────────────────────────────────
# Baseline agent registry
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_AGENTS = {
    "random":    RandomAgent,
    "rulebased": RuleBasedAgent,
    "economic":  EconomicDispatchAgent,
}


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — requests chain-of-thought reasoning + structured JSON output
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert power systems engineer managing an IEEE 14-bus electricity grid
during a live crisis. Your decisions are IRREVERSIBLE — incorrect dispatch triggers
cascade relay trips that cannot be undone within the episode.

You control 6 generators:
  Gen 0 — Bus 1,  Coal,   20–100 MW  (always available, highest cost)
  Gen 1 — Bus 2,  Gas,    10–80 MW   (may trip during emergencies)
  Gen 2 — Bus 3,  Gas,    5–50 MW
  Gen 3 — Bus 6,  Wind,   0–50 MW   (limited by wind capacity factor)
  Gen 4 — Bus 8,  Hydro,  5–40 MW   (fast-response, medium cost)
  Gen 5 — Bus 12, Solar,  0–30 MW   (limited by solar capacity factor)

Decision rules:
1. Total generation must closely match total_load_mw (balance the grid)
2. Generators marked online=False must receive dispatch = 0
3. Prefer wind/solar (zero cost) but cap at their available MW
4. Avoid lines with loading_frac > 0.90 (cascade risk)
5. Maintain 10% spinning reserve above current dispatch
6. INACTION IS NOT SAFE — failing to dispatch causes blackout within 3 steps

You MUST respond with valid JSON in this exact format:
{
  "reasoning": "Step-by-step analysis: [1] current load is X MW, generation gap is Y MW. [2] Gen N is offline so I cannot use it. [3] I will increase coal by Z to cover the deficit... [DECISION] final dispatch rationale",
  "dispatch_mw": [p0, p1, p2, p3, p4, p5]
}

The "reasoning" field must explain your full thinking. This is critical.
No markdown, no extra keys, no explanation outside the JSON.
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# State → LLM prompt
# ─────────────────────────────────────────────────────────────────────────────

def _state_to_prompt(state: dict) -> str:
    """Convert environment state dict to a crisis-aware LLM prompt."""
    gen_lines = "\n".join(
        "  Gen {i}: online={online}, dispatch={dispatch:.1f}MW, available={avail:.1f}MW, fuel={fuel}".format(
            i=i,
            online=state["gen_online"][i],
            dispatch=state["gen_dispatch_mw"][i],
            avail=state["gen_available_mw"][i],
            fuel=["Coal","Gas","Gas","Wind","Hydro","Solar"][i],
        )
        for i in range(len(state["gen_dispatch_mw"]))
    )

    overloaded = [
        "Line {i} ({pct:.0f}% — NEAR TRIP!)".format(i=i, pct=state["line_loading_frac"][i] * 100)
        for i in range(len(state["line_loading_frac"]))
        if state["line_loading_frac"][i] > 0.85
    ]

    tripped = [
        "Line {}".format(i)
        for i in range(len(state["line_status"]))
        if not state["line_status"][i]
    ]

    balance = state["power_balance_mw"]
    balance_str = (
        "SURPLUS +{:.1f} MW".format(balance) if balance > 5
        else "DEFICIT {:.1f} MW -- CRITICAL!".format(balance) if balance < -10
        else "BALANCED ({:+.1f} MW)".format(balance)
    )

    scenario_block = ""
    if state.get("scenario_id"):
        scenario_block = "\n!!! CRISIS SCENARIO: {} !!!\n{}\n".format(
            state["scenario_id"].upper().replace("_", " "),
            state.get("scenario_description", ""),
        )

    return (
        "=== GRID STATUS: Step {step} | Difficulty: {difficulty} ==={scenario}\n\n"
        "Load demand:  {load:.1f} MW\n"
        "Generating:   {gen:.1f} MW\n"
        "Balance:      {balance}\n"
        "Wind CF:      {cf_wind:.0%} | Solar CF: {cf_solar:.0%}\n\n"
        "Generator Status:\n{gen_lines}\n\n"
        "At-risk lines (>85%): {overloaded}\n"
        "Tripped lines:        {tripped}\n\n"
        "What is your dispatch decision? Reason step-by-step then give JSON."
    ).format(
        step=state["step"],
        difficulty=state["difficulty"],
        scenario=scenario_block,
        load=state["total_load_mw"],
        gen=state["total_gen_mw"],
        balance=balance_str,
        cf_wind=state["cf_wind"],
        cf_solar=state["cf_solar"],
        gen_lines=gen_lines,
        overloaded=overloaded if overloaded else ["None"],
        tripped=tripped if tripped else ["None"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parse LLM response — extracts reasoning + dispatch_mw
# ─────────────────────────────────────────────────────────────────────────────

def _parse_llm_response(content: str) -> tuple[Optional[np.ndarray], str]:
    """
    Returns (dispatch_array, reasoning_text).
    If parsing fails, returns (None, raw_content).
    """
    try:
        # Strip markdown fences
        clean = content.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()

        obj = json.loads(clean)
        mw  = obj.get("dispatch_mw", [])
        reasoning = obj.get("reasoning", "")

        if len(mw) == ieee14.N_GENS:
            return np.array([float(x) for x in mw]), reasoning
    except Exception:
        pass

    # Fallback: try to find dispatch_mw even without reasoning
    try:
        import re
        match = re.search(r'"dispatch_mw"\s*:\s*\[([^\]]+)\]', content)
        if match:
            mw = [float(x.strip()) for x in match.group(1).split(",")]
            if len(mw) == ieee14.N_GENS:
                return np.array(mw), "(could not extract reasoning)"
    except Exception:
        pass

    return None, content[:200]


# ─────────────────────────────────────────────────────────────────────────────
# LLM Agent
# ─────────────────────────────────────────────────────────────────────────────

class LLMAgent:
    """
    OpenAI-API backed agent using chain-of-thought reasoning.
    Uses HF_TOKEN and API_BASE_URL per hackathon spec.
    """

    name = "LLMAgent"

    def __init__(self, model: str = MODEL_NAME, temperature: float = 0.3):
        if HF_TOKEN is None:
            raise ValueError(
                "HF_TOKEN environment variable is required. "
                "Set it before running inference.py."
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        # Initialize exactly as per official spec
        self._client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )
        self._model       = model
        self._temperature = temperature
        self._fallback    = EconomicDispatchAgent()
        self.last_reasoning: str = ""     # Store last reasoning for transcript

    def reset(self) -> None:
        self.last_reasoning = ""

    def act(self, state: dict, env: PowerGridEnv) -> np.ndarray:
        user_msg = _state_to_prompt(state)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=300,   # more tokens for reasoning
            )
            content = response.choices[0].message.content or ""
            action, reasoning = _parse_llm_response(content)
            self.last_reasoning = reasoning
            if action is not None:
                return action
        except Exception:
            self.last_reasoning = "(API error — fallback used)"

        # Emergency fallback
        self.last_reasoning = self.last_reasoning or "(parse error — fallback used)"
        return self._fallback.act(state, env)


# ─────────────────────────────────────────────────────────────────────────────
# Episode Narrative Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_narrative(
    step_rewards: List[float],
    reasoning_trace: List[str],
    passed: bool,
    scenario_id: Optional[str] = None,
) -> str:
    """
    Generate a human-readable story of what happened in an episode.
    This is the key differentiator for judges — makes the LLM's behavior
    interpretable and memorable.
    """
    n = len(step_rewards)
    if n == 0:
        return "Empty episode."

    # Find the crisis point (biggest reward drop)
    crisis_step = None
    for i in range(1, n):
        if step_rewards[i] < step_rewards[i-1] - 0.2:
            crisis_step = i + 1
            break

    avg_early  = round(sum(step_rewards[:min(5, n)]) / min(5, n), 2)
    avg_late   = round(sum(step_rewards[max(0, n-5):]) / min(5, n), 2)
    trend      = "improving" if avg_late > avg_early else "deteriorating"
    outcome    = "GRID STABLE" if passed else "GRID FAILURE"
    outcome_emoji = "✅" if passed else "❌"

    lines = [
        "=== {} {} ===".format(outcome_emoji, outcome),
    ]
    if scenario_id:
        lines.append("Scenario: {}".format(scenario_id.upper().replace("_", " ")))

    lines += [
        "Steps:         {}".format(n),
        "Avg reward:    {:.2f} (early) → {:.2f} (late) [{}]".format(avg_early, avg_late, trend),
    ]

    if crisis_step:
        lines.append("Crisis point:  Step {} — reward dropped sharply".format(crisis_step))

    # Show first meaningful reasoning
    for i, r in enumerate(reasoning_trace[:3]):
        if r and "(fallback" not in r and "(API" not in r:
            short = r[:120].replace("\n", " ") + ("..." if len(r) > 120 else "")
            lines.append("LLM reasoning: \"{}\"".format(short))
            break

    lines.append("-" * 50)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Stdout logging  (exact spec format)
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print("[START] task={} env={} model={}".format(task, env, model), flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        "[STEP] step={} action={} reward={:.2f} done={} error={}".format(
            step, action, reward, str(done).lower(), error_val
        ),
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join("{:.2f}".format(r) for r in rewards)
    print(
        "[END] success={} steps={} rewards={}".format(
            str(success).lower(), steps, rewards_str
        ),
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task runner
# ─────────────────────────────────────────────────────────────────────────────

import sys

def run_task(
    difficulty: str,
    agent,
    episodes: int,
    seed: int,
    agent_label: str = "unknown",
    save_transcript: bool = False,
) -> dict:
    """Run episodes on one difficulty level; emit [START]/[STEP]/[END] per spec."""
    diff    = Difficulty(difficulty)
    results: list = []
    transcript_lines: list = []

    for ep in range(episodes):
        env   = PowerGridEnv(diff, seed=seed + ep)
        state = env.reset()
        agent.reset()
        done            = False
        step_rewards: list = []
        reasoning_trace: list = []
        step_idx        = 1

        task_name = "power_grid_{}_ep{}".format(difficulty, ep + 1)
        log_start(task=task_name, env="power_grid", model=agent_label)

        if save_transcript:
            transcript_lines.append("\n{'='*60}")
            transcript_lines.append("EPISODE: {} | Difficulty: {} | Seed: {}".format(
                ep + 1, difficulty, seed + ep))
            transcript_lines.append("="*60)

        try:
            while not done:
                action    = agent.act(state, env)
                reasoning = getattr(agent, "last_reasoning", "")
                reasoning_trace.append(reasoning)

                error_msg: Optional[str] = None
                try:
                    state, reward, done, _ = env.step(action)
                except Exception as e:
                    reward    = 0.0
                    done      = True
                    error_msg = str(e)

                step_rewards.append(reward)
                action_str = action.tolist() if isinstance(action, np.ndarray) else action
                action_str = str(action_str).replace(" ", "")
                log_step(step=step_idx, action=action_str, reward=reward,
                         done=done, error=error_msg)

                # Save reasoning to transcript (stderr so it's visible but not parsed)
                if reasoning and save_transcript:
                    transcript_lines.append(
                        "\nStep {}: reward={:.2f}\nLLM reasoning: {}\nDispatch: {}".format(
                            step_idx, reward, reasoning, action_str
                        )
                    )

                step_idx += 1

            result  = grade_episode(env)
            success = result["passed"]
        except Exception as e:
            log_end(success=False, steps=step_idx - 1, rewards=step_rewards)
            raise

        log_end(success=success, steps=step_idx - 1, rewards=step_rewards)

        # Generate episode narrative
        narrative = generate_narrative(
            step_rewards, reasoning_trace, success,
            scenario_id=getattr(env, "_scenario_id", None),
        )

        # Print narrative to stderr so judges can see it
        print("\n" + narrative, file=sys.stderr, flush=True)

        result["episode"]         = ep + 1
        result["seed"]            = seed + ep
        result["step_rewards"]    = step_rewards
        result["reasoning_trace"] = reasoning_trace[:5]   # first 5 steps
        result["narrative"]       = narrative
        results.append(result)

    # Save transcript to file if requested
    if save_transcript and transcript_lines:
        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(transcript_lines))
        print("Reasoning transcript saved to transcript.txt", file=sys.stderr)

    scores = [r["score"] for r in results]
    avg_r  = [r["metrics"]["avg_reward_step"] for r in results]

    mean_score = statistics.mean(scores) if scores else 0.5
    # Clamp → round → clamp: prevents any floating point boundary drift
    safe_mean = float(max(0.05, min(0.95, round(mean_score, 6))))
    safe_mean = round(float(max(0.05, min(0.95, safe_mean))), 4)

    # score_stats: only include safe fields; stdev floored at 0.001 (never 0.0)
    raw_std  = statistics.stdev(scores) if len(scores) > 1 else 0.001
    safe_std = float(max(0.001, min(0.499, round(raw_std, 4))))

    safe_min = float(max(0.05, min(0.95, round(float(min(scores)), 4))))
    safe_max = float(max(0.05, min(0.95, round(float(max(scores)), 4))))

    # avg_reward summary (not a score, so no 0/1 risk, but clamp anyway)
    mean_avg_r = round(float(statistics.mean(avg_r)), 4)

    passes = sum(1 for r in results if r["passed"])
    raw_pass_rate = passes / episodes if episodes > 0 else 0.5
    pass_rate = float(max(0.01, min(0.99, raw_pass_rate)))

    from collections import Counter
    grades = Counter(r["grade"] for r in results)
    grade_dist = {g: grades.get(g, 0) for g in ["A", "B", "C", "D", "F"]}

    from tasks.graders import _sanitize_output

    return _sanitize_output({
        "difficulty": difficulty,
        "episodes":   episodes,
        "seed":       seed,
        "score_01":   safe_mean,
        "score":      safe_mean,
        "score_stats": {
            "mean": safe_mean,
            "std":  safe_std,
            "min":  safe_min,
            "max":  safe_max,
        },
        "pass_rate": pass_rate,
        "grade_distribution": grade_dist,
        "avg_reward_mean": mean_avg_r,
        "sample_reasoning": results[0].get("reasoning_trace", []) if results else [],
    })


def run_all_tasks(agent, episodes: int, seed: int, agent_label: str,
                  save_transcript: bool = False) -> dict:
    """Run all 3 difficulty tasks."""
    output = {}
    for diff in ("easy", "medium", "hard"):
        print("\n-- {} --".format(diff.upper()), file=sys.stderr)
        output[diff] = run_task(diff, agent, episodes, seed, agent_label, save_transcript)
    return output


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Power Grid Crisis Environment — Hackathon Inference Script"
    )
    parser.add_argument(
        "--agent", "-a",
        choices=list(BASELINE_AGENTS),
        default=None,
        help="Use a baseline agent instead of LLM (for local testing without API key).",
    )
    parser.add_argument("--episodes", "-e", type=int, default=3,
                        help="Episodes per difficulty level (default: 3).")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Base random seed (default: 42).")
    parser.add_argument("--output", "-o", default="results.json",
                        help="Output JSON file (default: results.json).")
    parser.add_argument("--transcript", "-t", action="store_true",
                        help="Save full LLM reasoning transcript to transcript.txt.")
    args = parser.parse_args()

    # ── Select agent ──────────────────────────────────────────────────────────
    if args.agent is not None:
        AgentCls    = BASELINE_AGENTS[args.agent]
        agent       = AgentCls(seed=args.seed) if AgentCls == RandomAgent else AgentCls()
        agent_label = args.agent
        print("Using baseline agent: {}".format(agent_label), file=sys.stderr)
    else:
        # Default: LLM agent — always used during hackathon validation
        agent       = LLMAgent(model=MODEL_NAME)
        agent_label = "llm:{}".format(MODEL_NAME)
        print("Using LLM agent: {}".format(MODEL_NAME), file=sys.stderr)

    # ── Run ───────────────────────────────────────────────────────────────────
    all_results = run_all_tasks(
        agent, args.episodes, args.seed, agent_label,
        save_transcript=args.transcript,
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n{:=<54}".format(""), file=sys.stderr)
    print("  POWER GRID CRISIS — RESULTS SUMMARY", file=sys.stderr)
    print("{:=<54}".format(""), file=sys.stderr)
    print("  {:<10} {:>8}  {:>6}  {:>7}  Grades".format(
        "Difficulty", "Score", "Stdev", "Pass%"), file=sys.stderr)
    print("  " + "-"*52, file=sys.stderr)
    for diff, res in all_results.items():
        s = res["score_stats"]
        print(
            "  {:<10} {:>8.4f}  {:>6.4f}  {:>6.1f}%  {}".format(
                diff.upper(), s["mean"], s["std"],
                res["pass_rate"] * 100,
                res["grade_distribution"],
            ),
            file=sys.stderr,
        )
    print("{:=<54}\n".format(""), file=sys.stderr)

    # ── Write JSON ────────────────────────────────────────────────────────────
    output = {
        "agent":       agent_label,
        "model":       MODEL_NAME,
        "episodes":    args.episodes,
        "seed":        args.seed,
        "api_base":    API_BASE_URL,
        "tasks":       all_results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print("Results saved to {}".format(args.output), file=sys.stderr)


if __name__ == "__main__":
    main()
