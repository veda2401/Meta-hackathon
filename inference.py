"""
inference.py
------------
Baseline inference script — OpenEnv Hackathon compliant.

Uses the OpenAI API client to run an LLM agent against the Power Grid
environment on all 3 difficulty levels, producing reproducible scores.

Usage
-----
    # LLM agent (requires OPENAI_API_KEY env variable)
    python inference.py --model gpt-4o-mini --episodes 3 --seed 42

    # Fallback baseline agents (no API key needed)
    python inference.py --agent economic --episodes 5 --seed 42

Environment variables
---------------------
    OPENAI_API_KEY   — OpenAI API key (required for --model flag)
    OPENAI_BASE_URL  — optional base URL override (e.g. for Azure / local)

Output
------
    results.json  — per-episode details + aggregate summary across all 3 tasks
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import textwrap
from typing import List, Optional

import numpy as np

from env.grid_env import PowerGridEnv, Difficulty
from env import ieee14
from agents.baselines import RandomAgent, RuleBasedAgent, EconomicDispatchAgent
from tasks.graders import grade_episode

# ─────────────────────────────────────────────────────────────────────────────
# Fallback agent registry (used when no LLM is available)
# ─────────────────────────────────────────────────────────────────────────────
BASELINE_AGENTS = {
    "random":    RandomAgent,
    "rulebased": RuleBasedAgent,
    "economic":  EconomicDispatchAgent,
}


# ─────────────────────────────────────────────────────────────────────────────
# LLM Agent  (OpenAI API client)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert power systems engineer controlling an IEEE 14-bus electricity grid.
Your goal is to dispatch generators to balance load, avoid line overloads, and
minimise fuel cost. You control 6 generators:

  Gen 0 — Bus 1,  Coal,   20–100 MW
  Gen 1 — Bus 2,  Gas,    10–80 MW
  Gen 2 — Bus 3,  Gas,    5–50 MW
  Gen 3 — Bus 6,  Wind,   0–50 MW  (weather-limited)
  Gen 4 — Bus 8,  Hydro,  5–40 MW
  Gen 5 — Bus 12, Solar,  0–30 MW  (weather-limited)

Rules:
- Total generation should closely match total_load_mw.
- Prefer wind and solar (lowest cost, zero emissions).
- Avoid lines with loading_frac > 0.90 (risk of overload).
- Keep 10% spinning reserve above dispatch.

Reply with ONLY a JSON object like:
{"dispatch_mw": [p0, p1, p2, p3, p4, p5]}

No explanation, no markdown, no extra keys.
""").strip()


def _state_to_prompt(state: dict) -> str:
    """Convert environment state dict to a concise LLM prompt."""
    gen_lines = "\n".join(
        f"  Gen {i}: online={state['gen_online'][i]}, "
        f"current={state['gen_dispatch_mw'][i]:.1f} MW, "
        f"available={state['gen_available_mw'][i]:.1f} MW"
        for i in range(len(state["gen_dispatch_mw"]))
    )
    overloaded = [
        f"Line {i} ({state['line_loading_frac'][i]*100:.0f}%)"
        for i in range(len(state["line_loading_frac"]))
        if state["line_loading_frac"][i] > 0.85
    ]
    return textwrap.dedent(f"""
Current grid state (step {state['step']}, difficulty: {state['difficulty']}):

Total load:       {state['total_load_mw']:.1f} MW
Total generation: {state['total_gen_mw']:.1f} MW
Power balance:    {state['power_balance_mw']:+.1f} MW
Wind CF:          {state['cf_wind']*100:.0f}%
Solar CF:         {state['cf_solar']*100:.0f}%

Generator status:
{gen_lines}

Warning lines (>85% loading): {overloaded if overloaded else 'None'}

What is your generator dispatch?
""").strip()


def _parse_llm_action(content: str) -> Optional[np.ndarray]:
    """Extract dispatch_mw list from LLM response JSON."""
    try:
        # Strip markdown fences if present
        content = content.strip().strip("```json").strip("```").strip()
        obj = json.loads(content)
        mw  = obj.get("dispatch_mw", [])
        if len(mw) == ieee14.N_GENS:
            return np.array([float(x) for x in mw])
    except Exception:
        pass
    return None


class LLMAgent:
    """OpenAI API-backed agent that reads grid state and dispatches generators."""

    name = "LLMAgent"

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        # Hackathon prerequisite variables
        api_key  = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("API_BASE_URL") or os.environ.get("OPENAI_BASE_URL")

        if not api_key:
            raise EnvironmentError(
                "HF_TOKEN or API_KEY is not set. "
                "Export your key or use --agent economic for a baseline run."
            )
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self._client      = OpenAI(**kwargs)
        self._model       = model or os.environ.get("MODEL_NAME", "gpt-4o-mini")
        self._temperature = temperature
        self._fallback    = EconomicDispatchAgent()

    def reset(self) -> None:
        pass

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
                max_tokens=64,
            )
            content = response.choices[0].message.content or ""
            action  = _parse_llm_action(content)
            if action is not None:
                return action
            print(f"  [LLM parse-error] raw='{content[:80]}' → falling back")
        except Exception as exc:
            print(f"  [LLM API error] {exc} → falling back")

        return self._fallback.act(state, env)


# ─────────────────────────────────────────────────────────────────────────────
# Core inference runner
# ─────────────────────────────────────────────────────────────────────────────

import sys

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def run_task(
    difficulty: str,
    agent,
    episodes: int,
    seed: int,
    agent_label: str = "unknown"
) -> dict:
    """Run `episodes` episodes on one difficulty level and return aggregate stats."""
    diff    = Difficulty(difficulty)
    results: list[dict] = []

    for ep in range(episodes):
        env   = PowerGridEnv(diff, seed=seed + ep)
        state = env.reset()
        agent.reset()
        done  = False
        step_rewards: list[float] = []

        task_name = f"power_grid_{difficulty}_ep{ep+1}"
        env_benchmark = "power_grid"
        log_start(task=task_name, env=env_benchmark, model=agent_label)

        step_idx = 1
        while not done:
            action = agent.act(state, env)
            try:
                state, reward, done, _ = env.step(action)
                error_msg = None
            except Exception as e:
                reward = 0.0
                done = True
                error_msg = str(e)
            
            step_rewards.append(reward)
            
            action_str = str(action.tolist()) if isinstance(action, np.ndarray) else str(action)
            action_str = action_str.replace(" ", "")

            log_step(step=step_idx, action=action_str, reward=reward, done=done, error=error_msg)
            step_idx += 1

        result = grade_episode(env)
        result["episode"]      = ep + 1
        result["seed"]         = seed + ep
        result["step_rewards"] = step_rewards
        results.append(result)

        log_end(success=result['passed'], steps=step_idx-1, score=result['score_01'], rewards=step_rewards)

    scores    = [r["score_01"] for r in results]
    avg_r     = [r["metrics"]["avg_reward/step"] for r in results]
    pass_rate = sum(1 for r in results if r["passed"]) / episodes

    return {
        "difficulty": difficulty,
        "episodes":   episodes,
        "seed":       seed,
        "score_01": {
            "mean": round(statistics.mean(scores), 4),
            "std":  round(statistics.stdev(scores) if episodes > 1 else 0.0, 4),
            "min":  round(min(scores), 4),
            "max":  round(max(scores), 4),
        },
        "avg_reward_per_step": {
            "mean": round(statistics.mean(avg_r), 4),
            "std":  round(statistics.stdev(avg_r) if episodes > 1 else 0.0, 4),
        },
        "pass_rate": round(pass_rate, 4),
        "grade_distribution": {
            g: sum(1 for r in results if r["grade"] == g) for g in "ABCDF"
        },
        "episodes_detail": results,
    }


def run_all_tasks(agent, episodes: int, seed: int, agent_label: str) -> dict:
    """Run all 3 difficulty tasks and produce a combined reproducible report."""
    tasks_output = {}
    for diff in ("easy", "medium", "hard"):
        print(f"\n── Task: {diff.upper()} ────────────────────────────────", file=sys.stderr)
        tasks_output[diff] = run_task(diff, agent, episodes, seed, agent_label)
    return tasks_output


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Power Grid OpenEnv — Reproducible Baseline Inference"
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="OpenAI model name (overrides MODEL_NAME env var). Requires HF_TOKEN or API_KEY.",
    )
    parser.add_argument(
        "--agent", "-a",
        choices=list(BASELINE_AGENTS), default="economic",
        help="Fallback baseline agent when --model is not set (default: economic).",
    )
    parser.add_argument("--episodes", "-e", type=int, default=3,
                        help="Episodes per difficulty level (default: 3).")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Base random seed for reproducibility (default: 42).")
    parser.add_argument("--output", "-o", default="results.json",
                        help="Output JSON file path (default: results.json).")
    args = parser.parse_args()

    # ── Select agent ──────────────────────────────────────────────────────────
    model_to_use = args.model or os.environ.get("MODEL_NAME")
    if model_to_use:
        print(f"\n⚡ Using LLM agent: {model_to_use}", file=sys.stderr)
        agent = LLMAgent(model=model_to_use)
        agent_label = f"llm:{model_to_use}"
    else:
        AgentCls = BASELINE_AGENTS[args.agent]
        agent    = AgentCls(seed=args.seed) if AgentCls == RandomAgent else AgentCls()
        agent_label = args.agent
        print(f"\n⚡ Using baseline agent: {agent_label}", file=sys.stderr)

    print(f"   Episodes per task: {args.episodes}  |  Seed: {args.seed}\n", file=sys.stderr)

    # ── Run all 3 tasks ───────────────────────────────────────────────────────
    all_results = run_all_tasks(agent, args.episodes, args.seed, agent_label)

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════╗", file=sys.stderr)
    print(  "║          REPRODUCIBLE BASELINE SCORES (0.0–1.0)     ║", file=sys.stderr)
    print(  "╠══════════════════════════════════════════════════════╣", file=sys.stderr)
    print(f"  {'Difficulty':<10} {'Score':>8}  {'±':>6}  {'Pass%':>7}  Grades", file=sys.stderr)
    print(  "  " + "─"*52, file=sys.stderr)
    for diff, res in all_results.items():
        s = res["score_01"]
        print(
            f"  {diff.upper():<10} {s['mean']:>8.4f}  "
            f"{s['std']:>6.4f}  "
            f"{res['pass_rate']*100:>6.1f}%  "
            f"{res['grade_distribution']}", file=sys.stderr
        )
    print("╚══════════════════════════════════════════════════════╝\n", file=sys.stderr)

    # ── Write JSON ────────────────────────────────────────────────────────────
    output = {
        "agent":    agent_label,
        "episodes": args.episodes,
        "seed":     args.seed,
        "tasks":    all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved → {args.output}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
