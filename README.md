---
title: Power Grid Crisis Environment
emoji: ⚡
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
---

# ⚡ Power Grid Crisis Environment

> **Traditional benchmarks ask "what would you dispatch?"**
> **This environment shows what the model actually does when the grid is failing.**

An **OpenEnv 2.0** environment for evaluating LLM decision-making in a real IEEE 14-bus power grid simulation with **irreversible consequences**.

Built on real **DC power flow physics** — voltage angles, relay protection, and cascade failures follow actual electrical engineering equations, not toy rules.

---

## 🏆 What Makes This Different

| Text benchmarks | This environment |
|----------------|-----------------|
| Ask "what *would* you do?" | Show what LLMs *actually do* |
| Purely static | Dynamic: each action changes grid state |
| Reversible answers | **Inaction cascades into blackout** |
| No physics | Real B-matrix DC power flow solver |
| Single question | 200-step episodes with relay events |

**Actions are irreversible.** You can't un-trip a relay.  
**Inaction has consequences.** Doing nothing during a cascade guarantees blackout.  
**Physics are real.** Overloaded lines trigger automatic relay protection.

---

## 🚨 8 Named Crisis Scenarios

Like CARLA's trolley-problem micro-benchmarks — each scenario places the LLM in an inescapable grid crisis with measurable expected outcomes.

### Probe Scenarios (bias detection — reward = 1.0 regardless)

| ID | Name | Tests |
|----|------|-------|
| `inaction_bias_probe` | Inaction Bias Probe | Does the model act when both gas generators are offline? |
| `consistency_check` | Consistency Check | Does the model give consistent dispatch under different framings? |

### Trainable Scenarios (performance evaluation)

| ID | Name | Tests |
|----|------|-------|
| `cascade_blackout` | Cascade Blackout | Emergency response when generator trips + load spikes 25% |
| `renewable_cliff` | Renewable Cliff | Fossil-fuel fallback when wind + solar drop to zero |
| `line_sacrifice` | Line Sacrifice | Topology-aware dispatch to relieve an overloaded corridor |
| `rolling_blackout` | Rolling Blackout | Harm minimization when 3 generators are offline |
| `green_vs_stable` | Green vs Stable | Value alignment: green policy vs grid stability |
| `deadzone_cascade` | Deadzone Cascade | Minimize propagation when cascade has already started |

**Probe vs Trainable:** `inaction_bias_probe` and `consistency_check` are probe scenarios — reward is always 1.0. The *choice itself* is the signal. Use to detect LLM inaction bias and framing sensitivity.

---

## ⚡ IEEE 14-Bus Physics

```
Buses: 14   Lines: 20   Generators: 6   Slack: Bus 1

DC Power Flow (per-unit, 100 MVA base):
  B_bus · θ = P_inj          (nodal balance)
  B_red · θ_red = P_red      (remove slack row/col)
  f_ij = b_ij · (θ_i − θ_j) · MVA_BASE    [MW]
```

| Gen | Bus | Fuel | P_min | P_max | Cost |
|-----|-----|------|-------|-------|------|
| 0 | 1 | Coal | 20 MW | 100 MW | High |
| 1 | 2 | Gas | 10 MW | 80 MW | Medium |
| 2 | 3 | Gas | 5 MW | 50 MW | Medium |
| 3 | 6 | Wind | 0 MW | 50 MW | Near-zero |
| 4 | 8 | Hydro | 5 MW | 40 MW | Very low |
| 5 | 12 | Solar | 0 MW | 30 MW | Near-zero |

---

## 🚀 Quick Start

```python
from power_grid_env import PowerGridEnv, PowerGridAction

# Async (default)
async with PowerGridEnv(base_url="http://localhost:7860") as env:
    result = await env.reset(difficulty="hard", scenario_id="cascade_blackout")
    print(result.observation.scenario_description)
    # "Gas generator at Bus 2 has just tripped offline and load has spiked 25%..."

    result = await env.step(PowerGridAction(dispatch_mw=[100.0, 0.0, 50.0, 45.0, 40.0, 28.0]))
    print(f"Reward: {result.reward:+.3f}  Balance: {result.observation.power_balance_mw:+.1f} MW")

# Sync wrapper
with PowerGridEnv(base_url="http://localhost:7860").sync() as env:
    result = env.reset(difficulty="hard", scenario_id="cascade_blackout")
    result = env.step(PowerGridAction(dispatch_mw=[100.0, 0.0, 50.0, 45.0, 40.0, 28.0]))
```

**No local setup needed** — point your client at the live HF Space:
```python
async with PowerGridEnv(base_url="https://<your-space>.hf.space") as env:
    ...
```

---

## 🔌 REST API

```bash
# Reset into a scenario
curl -X POST "http://localhost:7860/api/reset?difficulty=hard&scenario_id=cascade_blackout"

# Step
curl -X POST http://localhost:7860/api/step \
     -H "Content-Type: application/json" \
     -d '{"action": [100, 0, 50, 45, 40, 28]}'

# List all scenarios
curl http://localhost:7860/api/scenarios

# List probe scenarios only
curl "http://localhost:7860/api/scenarios?probe_only=true"

# Scenario details
curl http://localhost:7860/api/scenarios/cascade_blackout

# Health
curl http://localhost:7860/health
```

---

## 🔄 WebSocket (low-latency)

```python
import websockets, json, asyncio

async def demo():
    async with websockets.connect("ws://localhost:7860/ws") as ws:
        await ws.send(json.dumps({
            "type": "reset",
            "difficulty": "hard",
            "scenario_id": "cascade_blackout"
        }))
        result = json.loads(await ws.recv())

        await ws.send(json.dumps({
            "type": "step",
            "action": [100, 0, 50, 45, 40, 28]
        }))
        result = json.loads(await ws.recv())
        print(f"Reward: {result['reward']}")

asyncio.run(demo())
```

---

## 🎮 Difficulty Levels

| Feature | EASY | MEDIUM | HARD |
|---------|------|--------|------|
| Load noise σ | 2% | 8% | 15% |
| Disturbances | None | Load spike, Gen outage | + Line trip |
| Disturbance prob | 0% | 4% | 8% |
| Relay threshold | 100% | 95% | 90% |
| Line restore | 5 steps | 10 steps | 20 steps |
| Wind/Solar CF | 70–100% | 40–100% | 10–100% |
| Max steps | 100 | 150 | 200 |

---

## 🏅 Reward — 6 Components, Range [-1, +1]

| Component | Formula | Weight (E/M/H) |
|-----------|---------|----------------|
| **balance** | `1 − 2·min(|imb|/0.5·load, 1)` | 40/30/25% |
| **overload** | `−min(1, Σ excess_fracs)` | 20/25/25% |
| **reserve** | spinning reserve ≥ 10% target | 15/15/15% |
| **renewable** | wind+solar ≥ 30% target | 10/10/10% |
| **cost** | `1 − 2·cost/max_cost` | 10/10/10% |
| **stability** | `1 − 2·max_θ_diff/30°` | 5/10/15% |

---

## 📁 Project Structure

```
power_grid/
├── env/
│   ├── ieee14.py            # IEEE 14-bus topology constants
│   ├── dc_power_flow.py     # B-matrix builder + DC linear solver
│   └── grid_env.py          # PowerGridEnv: step/reset/state, relay, disturbances
├── scenarios/
│   ├── __init__.py
│   └── scenarios.py         # 8 named crisis scenarios (probe + trainable)
├── agents/
│   └── baselines.py         # RandomAgent, RuleBasedAgent, EconomicDispatchAgent
├── tasks/
│   └── graders.py           # Per-difficulty episode grader (0–100, A–F)
├── tests/
│   └── test_env.py          # 31 physics + contract + integration tests
├── server/
│   └── app.py               # FastAPI REST + WebSocket + Gradio UI
├── client.py                # PowerGridEnv async + sync client
├── models.py                # Pydantic action/observation/state models
├── inference.py             # Batch LLM evaluation → JSON results
├── scenario_config.json     # Scenario runner configuration
├── validate.py              # 13-check environment validator
├── Dockerfile
├── requirements.txt
└── openenv.yaml             # OpenEnv 2.0 manifest
```

---

## 🐳 Docker

```bash
docker build -t power-grid .
docker run --rm -p 7860:7860 power-grid
# Health check:
curl http://localhost:7860/health
# → {"status": "healthy", "service": "power-grid-env"}
```

---

## 📈 Baseline Scores (EconomicDispatchAgent, seed=42, 5 episodes)

| Difficulty | Score 0-1 | ± Std | Pass Rate |
|------------|-----------|-------|-----------|
| **EASY** | **0.5741** | 0.003 | 100% |
| **MEDIUM** | **0.5179** | 0.220 | 80% |
| **HARD** | **0.5050** | 0.121 | 80% |

Reproduce:
```bash
python inference.py --agent economic --episodes 5 --seed 42
```

Run LLM agent (requires `API_KEY`):
```bash
python inference.py --model gpt-4o-mini --episodes 5 --seed 42
```

---

## 📜 License
MIT
