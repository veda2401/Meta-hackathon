---
title: Meta Hackathon
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
# ⚡ Power Grid Optimization Environment

> **Competition-grade** Agentic AI benchmark — IEEE 14-bus, real DC power flow,
> 6 generators × 5 fuel types, dense shaped reward, relay protection,
> FastAPI + Gradio on HF Spaces.

---

## 🏆 Technical Highlights

| Feature | Details |
|---------|---------|
| **DC Power Flow** | Real B-matrix linear solver (`numpy.linalg.solve`, 13×13 reduced system) — not a toy simulation |
| **Topology** | IEEE 14-bus standard test case: 14 buses, 20 transmission lines |
| **Generators** | 6 generators across 5 fuel types (Coal, Gas×2, Wind, Hydro, Solar) |
| **Reward** | Dense shaped, 6 components, guaranteed range **[-1, +1]** |
| **Stochasticity** | Gaussian load noise + 3 disturbance event types on MEDIUM / HARD |
| **Relay Protection** | Automatically trips lines at `threshold × rating`; blocks switching overloaded lines |
| **OpenEnv spec** | Full compliance: `step()`, `reset()`, `state()`, `openenv.yaml` |
| **Web** | FastAPI REST + Gradio UI in one `app.py` — HF Spaces ready |

---

## 📁 Project Structure

```
power_grid/
├── env/
│   ├── ieee14.py          # IEEE 14-bus topology constants (pure data)
│   ├── dc_power_flow.py   # B-matrix builder + DC linear solver
│   └── grid_env.py        # PowerGridEnv: step/reset/state, relay, disturbances
├── agents/
│   └── baselines.py       # RandomAgent, RuleBasedAgent, EconomicDispatchAgent
├── tasks/
│   └── graders.py         # Per-difficulty episode grader (0–100, A–F)
├── tests/
│   └── test_env.py        # 31 physics + contract + integration tests
├── app.py                 # FastAPI REST + Gradio UI (HF Spaces)
├── inference.py           # Batch evaluation → JSON
├── validate.py            # 13-check environment validator (CI-ready)
├── Dockerfile
├── requirements.txt
├── openenv.yaml           # OpenEnv 0.2 full spec
└── README.md
```

---

## ⚡ IEEE 14-Bus Physics

```
Buses: 14   Lines: 20   Generators: 6   Slack: Bus 1

DC Power Flow equations (per-unit, 100 MVA base):
  B_bus · θ = P_inj          (nodal balance)
  B_red · θ_red = P_red      (remove slack row/col)
  f_ij = b_ij · (θ_i − θ_j) · MVA_BASE    [MW]
```

Generator fleet:

| ID | Bus | Fuel | P_min | P_max | Cost ($/MWh) |
|----|-----|------|-------|-------|--------------|
| 0  | 1   | Coal | 20 MW | 100 MW | High |
| 1  | 2   | Gas  | 10 MW | 80 MW  | Medium |
| 2  | 3   | Gas  | 5 MW  | 50 MW  | Medium |
| 3  | 6   | Wind | 0 MW  | 50 MW  | Near-zero |
| 4  | 8   | Hydro| 5 MW  | 40 MW  | Very low |
| 5  | 12  | Solar| 0 MW  | 30 MW  | Near-zero |

---

## 🎮 Difficulty Levels

| Feature | EASY | MEDIUM | HARD |
|---------|------|--------|------|
| Load noise σ | 2 % | 8 % | 15 % |
| Disturbances | None | Load spike, Gen outage | + Line trip |
| Disturbance prob | 0 % | 4 % | 8 % |
| Relay threshold | 100 % | 95 % | 90 % |
| Line restore | 5 steps | 10 steps | 20 steps |
| Wind/Solar CF range | 70–100 % | 40–100 % | 10–100 % |
| Max steps | 100 | 150 | 200 |

---

## 🏅 Reward — 6 Components, Range [-1, +1]

| Component | Formula | Weight (E/M/H) |
|-----------|---------|---------------|
| **balance** | `1 − 2·min(|imb|/0.5·load, 1)` | 40/30/25 % |
| **overload** | `−min(1, Σ excess_fracs)` | 20/25/25 % |
| **reserve** | spinning reserve vs 10 % target | 15/15/15 % |
| **renewable** | wind+solar vs 30 % target | 10/10/10 % |
| **cost** | `1 − 2·cost/max_cost` | 10/10/10 % |
| **stability** | `1 − 2·max_θ_diff/30°` | 5/10/15 % |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
```

**Validate (CI check):**
```bash
python validate.py        # 13 physics + contract checks
```

**Run tests:**
```bash
python -m pytest tests/ -v   # 31 tests
```

**Interactive CLI:**
```bash
python app.py           # Gradio UI on http://localhost:7860
```

**REST API:**
```bash
# Reset
curl -X POST "http://localhost:7860/api/reset?difficulty=hard"

# Step
curl -X POST http://localhost:7860/api/step \
     -H "Content-Type: application/json" \
     -d '{"action": [80, 60, 40, 45, 35, 25]}'

# State
curl http://localhost:7860/api/state

# Topology info
curl http://localhost:7860/api/info
```

---

## 🐳 Docker

```bash
docker build -t power-grid .
docker run --rm -p 7860:7860 power-grid
```

---

## 🔌 Python API

```python
from env.grid_env import PowerGridEnv, Difficulty
from agents.baselines import EconomicDispatchAgent
from tasks.graders import grade_episode
import numpy as np

env   = PowerGridEnv(Difficulty.HARD, seed=42)
agent = EconomicDispatchAgent()
state = env.reset()
done  = False

while not done:
    action = agent.act(state, env)
    state, reward, done, info = env.step(action)
    print(f"Step {info['step']:>3}  R={reward:>+.3f}  "
          f"ΔP={state['power_balance_mw']:>+6.1f} MW  "
          f"Relay: {info['relay_tripped']}")

result = grade_episode(env)
print(f"\nGrade: {result['grade']}  ({result['total_score']:.1f}/100)")
print(f"Reward components: {result['avg_reward_components']}")
```

---

## 📈 Baseline Scores (EconomicDispatchAgent, seed=42, 3 episodes)

| Difficulty | Score 0.0–1.0 | ± Std | Pass Rate |
|------------|--------------|-------|----------|
| **EASY**   | **0.4402**   | 0.003 | 100 %     |
| **MEDIUM** | **0.3979**   | 0.112 | 67 %      |
| **HARD**   | **0.2995**   | 0.082 | 33 %      |

Reproduce with:
```bash
python inference.py --agent economic --episodes 3 --seed 42
```

Run LLM agent (requires `OPENAI_API_KEY`):
```bash
export OPENAI_API_KEY=sk-...
python inference.py --model gpt-4o-mini --episodes 3 --seed 42
```

---

## 📜 License
MIT
