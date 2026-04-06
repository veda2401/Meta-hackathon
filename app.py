"""
app.py
------
Power Grid Optimization — FastAPI REST API + Gradio UI
Designed for Hugging Face Spaces (port 7860).

Architecture
------------
  FastAPI app  →  REST endpoints at /api/*
  Gradio Blocks UI mounted at /  via gr.mount_gradio_app

Usage
-----
  python app.py                        # launches on http://0.0.0.0:7860
  python app.py --host 127.0.0.1 --port 8080
"""

from __future__ import annotations

import argparse
import json
import threading
from typing import List, Optional

import numpy as np

# ── FastAPI ─────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Gradio ──────────────────────────────────────────────────────────────────
import gradio as gr

# ── Project ─────────────────────────────────────────────────────────────────
from env.grid_env import PowerGridEnv, Difficulty
from env import ieee14
from agents.baselines import RandomAgent, RuleBasedAgent, EconomicDispatchAgent
from tasks.graders import grade_episode


# ═══════════════════════════════════════════════════════════════════════════
# Global session state (shared between REST API and Gradio UI)
# ═══════════════════════════════════════════════════════════════════════════

_lock = threading.Lock()

AGENT_REGISTRY = {
    "random":   lambda: RandomAgent(seed=42),
    "rulebased": RuleBasedAgent,
    "economic": EconomicDispatchAgent,
}

_env:   Optional[PowerGridEnv] = None
_agent: Optional[object]       = None
_last_state: dict              = {}
_step_rewards: List[float]     = []


def _get_env() -> PowerGridEnv:
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /api/reset first.")
    return _env


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI application + endpoints
# ═══════════════════════════════════════════════════════════════════════════

fastapi_app = FastAPI(
    title="Power Grid Optimization API",
    description="IEEE 14-Bus DC Power Flow Environment with OpenEnv compliance.",
    version="1.0.0",
)


class StepRequest(BaseModel):
    action: List[float]


@fastapi_app.post("/api/reset")
@fastapi_app.post("/reset")
def api_reset(difficulty: str = "easy"):
    """Reset environment and return initial state."""
    global _env, _last_state, _step_rewards
    if difficulty not in [d.value for d in Difficulty]:
        raise HTTPException(status_code=422,
                            detail=f"Unknown difficulty '{difficulty}'")
    with _lock:
        _env   = PowerGridEnv(Difficulty(difficulty), seed=None)
        state  = _env.reset()
        _last_state   = state
        _step_rewards = []
    # Note: openenv validate spec expects returning the state natively or via JSON
    return JSONResponse({"state": state, "message": f"Environment reset ({difficulty})"})


@fastapi_app.post("/api/step")
def api_step(req: StepRequest):
    """Advance one step with the provided generator dispatch action."""
    env = _get_env()
    if len(req.action) != ieee14.N_GENS:
        raise HTTPException(status_code=422,
                            detail=f"Action must have {ieee14.N_GENS} values.")
    with _lock:
        state, reward, done, info = env.step(np.array(req.action))
        _last_state.update(state)
        _step_rewards.append(reward)
    return JSONResponse({"state": state, "reward": reward, "done": done, "info": info})


@fastapi_app.get("/api/state")
def api_state():
    """Return current environment state without advancing."""
    return JSONResponse(_get_env().state())


@fastapi_app.get("/api/info")
def api_info():
    """Return static environment topology and generator metadata."""
    return JSONResponse({
        "n_buses":    ieee14.N_BUSES,
        "n_lines":    ieee14.N_LINES,
        "n_gens":     ieee14.N_GENS,
        "mva_base":   ieee14.MVA_BASE,
        "generators": [
            {"id": i, "bus": ieee14.GEN_BUS_IDX[i] + 1,
             "fuel": ieee14.GEN_FUEL[i],
             "p_min": ieee14.GEN_P_MIN[i], "p_max": ieee14.GEN_P_MAX[i]}
            for i in range(ieee14.N_GENS)
        ],
        "lines": [
            {"id": k, "from": ieee14.LINE_FROM[k] + 1,
             "to": ieee14.LINE_TO[k] + 1,
             "rating_mw": ieee14.LINE_RATINGS_MW[k]}
            for k in range(ieee14.N_LINES)
        ],
    })


@fastapi_app.post("/api/agent_step")
def api_agent_step(agent_name: str = "economic"):
    """Let the chosen baseline agent take one step."""
    global _agent
    env = _get_env()
    with _lock:
        if _agent is None or not isinstance(_agent, AGENT_REGISTRY.get(agent_name, type(None))):
            _agent = AGENT_REGISTRY[agent_name]()
        state  = env.state()
        action = _agent.act(state, env)
        state, reward, done, info = env.step(action)
        _last_state.update(state)
        _step_rewards.append(reward)
    return JSONResponse({"action": action.tolist(), "state": state,
                         "reward": reward, "done": done, "info": info})


# ═══════════════════════════════════════════════════════════════════════════
# Plotly grid visualisation helper
# ═══════════════════════════════════════════════════════════════════════════

def _make_grid_figure(state: dict) -> "go.Figure":
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    flows   = state.get("line_flows_mw", [0.0] * ieee14.N_LINES)
    loading = state.get("line_loading_frac", [0.0] * ieee14.N_LINES)
    lstat   = state.get("line_status", [True] * ieee14.N_LINES)
    gen_mw  = state.get("gen_dispatch_mw", [0.0] * ieee14.N_GENS)
    gen_on  = state.get("gen_online", [True] * ieee14.N_GENS)
    loads   = state.get("bus_loads_mw", ieee14.BASE_LOAD_MW)

    xs = [xy[0] for xy in ieee14.BUS_XY]
    ys = [xy[1] for xy in ieee14.BUS_XY]

    # -- Lines --
    line_traces = []
    for k in range(ieee14.N_LINES):
        fi, ti = ieee14.LINE_FROM[k], ieee14.LINE_TO[k]
        lf = loading[k]
        if not lstat[k]:
            colour = "gray"
            dash   = "dash"
        elif lf < 0.75:
            colour = f"rgb(0,{int(200*(1-lf/0.75))},0)"
        elif lf < 1.0:
            colour = f"rgb({int(255*(lf-0.75)/0.25)},165,0)"
        else:
            colour = "red"
            dash   = "solid"
        dash = "dash" if not lstat[k] else "solid"
        line_traces.append(go.Scatter(
            x=[xs[fi], xs[ti], None], y=[ys[fi], ys[ti], None],
            mode="lines",
            line=dict(color=colour, width=max(1, int(lf * 4)), dash=dash),
            hoverinfo="text",
            text=f"Line {fi+1}→{ti+1}<br>Flow: {flows[k]:.1f} MW<br>Loading: {lf*100:.0f}%",
            showlegend=False,
        ))

    # -- Bus nodes --
    colours, symbols, sizes, hover_text = [], [], [], []
    for i in range(ieee14.N_BUSES):
        gen_idx = next((g for g in range(ieee14.N_GENS) if ieee14.GEN_BUS_IDX[g] == i), None)
        if i == ieee14.SLACK_IDX:
            colours.append("#FFD700"); symbols.append("star"); sizes.append(22)
            hover_text.append(f"Bus 1 (Slack/Coal)<br>Load: {loads[i]:.1f} MW")
        elif gen_idx is not None:
            fuel   = ieee14.GEN_FUEL[gen_idx]
            online = gen_on[gen_idx]
            clr    = {"coal":"#555","gas":"#4488FF","wind":"#00CC77",
                      "hydro":"#00AAFF","solar":"#FFAA00"}.get(fuel,"#888")
            colours.append(clr if online else "#ccc")
            symbols.append("circle"); sizes.append(18)
            hover_text.append(
                f"Bus {i+1} ({fuel.title()})<br>"
                f"Gen: {gen_mw[gen_idx]:.1f} MW<br>"
                f"Load: {loads[i]:.1f} MW"
            )
        else:
            colours.append("#2255CC"); symbols.append("circle"); sizes.append(12)
            hover_text.append(f"Bus {i+1}<br>Load: {loads[i]:.1f} MW")

    node_trace = go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(color=colours, size=sizes, symbol=symbols,
                    line=dict(color="white", width=1.5)),
        text=[f"{i+1}" for i in range(ieee14.N_BUSES)],
        textposition="middle center",
        textfont=dict(color="white", size=9),
        hovertext=hover_text, hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=[*line_traces, node_trace])
    fig.update_layout(
        title=dict(text="⚡ IEEE 14-Bus Grid", font=dict(size=14, color="#eee")),
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        autosize=True,
    )
    return fig


def _make_reward_bar(components: dict, difficulty: str = "easy") -> "go.Figure":
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    BENCHMARKS = {
        "easy": {
            "balance": 0.9204, "overload": 0.0, "reserve": 1.0,
            "renewable": 0.6886, "cost": -0.3509, "stability": 0.9181
        },
        "medium": {
            "balance": 0.2129, "overload": 0.0, "reserve": -0.5115,
            "renewable": -0.0032, "cost": -0.2399, "stability": 0.8374
        },
        "hard": {
            "balance": -0.2528, "overload": 0.0, "reserve": -0.4720,
            "renewable": -0.3502, "cost": 0.0299, "stability": 0.8794
        }
    }

    keys = ["balance", "overload", "reserve", "renewable", "cost", "stability"]
    vals = [components.get(k, 0.0) for k in keys]
    colours = ["#00cc77" if v >= 0 else "#ff4455" for v in vals]
    bench_vals = [BENCHMARKS[difficulty][k] for k in keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=keys, y=vals, marker_color=colours,
        text=[f"{v:.2f}" for v in vals], textposition="outside",
        name="Current"
    ))
    fig.add_trace(go.Bar(
        x=keys, y=bench_vals, marker_color="#555",
        text=[f"{v:.2f}" for v in bench_vals], textposition="outside",
        name="Benchmark"
    ))
    fig.update_layout(
        barmode="group",
        title=dict(text="Reward Components vs Baseline (" + difficulty.upper() + ")", font=dict(size=13, color="#eee")),
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#ccc"),
        yaxis=dict(range=[-1.1, 1.1], gridcolor="#333"),
        xaxis=dict(gridcolor="#333"),
        margin=dict(l=20, r=20, t=50, b=20),
        autosize=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# =============================================================================
# Gradio Blocks UI
# =============================================================================

def _make_reward_history_plot(step_rewards: list) -> "go.Figure":
    """Line chart of reward per step across an episode."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    if not step_rewards:
        return None
    steps = list(range(1, len(step_rewards) + 1))
    colours = ["#00cc77" if r >= 0 else "#ff4455" for r in step_rewards]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=step_rewards, mode="lines+markers",
        line=dict(color="#4488ff", width=2),
        marker=dict(color=colours, size=5),
        name="Reward",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)
    fig.update_layout(
        title=dict(text="Reward per Step", font=dict(size=13, color="#eee")),
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#ccc"),
        xaxis=dict(title="Step", gridcolor="#333"),
        yaxis=dict(title="Reward", range=[-1.1, 1.1], gridcolor="#333"),
        margin=dict(l=30, r=20, t=40, b=30),
        autosize=True,
    )
    return fig


with gr.Blocks(title="Power Grid Optimizer") as demo:
    gr.Markdown(
        "# Power Grid Optimization Environment\n"
        "**IEEE 14-Bus | DC Power Flow | 6 Generators x 5 Fuels | 3 Difficulty Levels**"
    )

    with gr.Tabs():

        # -- Tab 1: Control Panel ------------------------------------------
        with gr.Tab("Control Panel"):
            with gr.Row():
                diff_dd  = gr.Dropdown(["easy", "medium", "hard"],
                                        value="easy", label="Difficulty")
                agent_dd = gr.Dropdown(["random", "rulebased", "economic"],
                                        value="economic", label="Baseline Agent")
                episodes = gr.Slider(1, 50, value=3, step=5, label="Episodes")

            with gr.Row():
                btn_reset = gr.Button("Reset",       variant="secondary")
                btn_step  = gr.Button("Agent Step",  variant="primary")
                btn_run   = gr.Button("Run Episode", variant="primary")

            with gr.Row():
                kpi_balance   = gr.Number(label="Balance Reward",   precision=3)
                kpi_overload  = gr.Number(label="Overload Reward",  precision=3)
                kpi_reserve   = gr.Number(label="Reserve Reward",   precision=3)
                kpi_renewable = gr.Number(label="Renewable Reward", precision=3)
                kpi_cost      = gr.Number(label="Cost Reward",      precision=3)
                kpi_stability = gr.Number(label="Stability Reward", precision=3)

            with gr.Row():
                kpi_total_gen  = gr.Number(label="Total Gen (MW)",     precision=1)
                kpi_total_load = gr.Number(label="Total Load (MW)",    precision=1)
                kpi_balance_mw = gr.Number(label="Balance Error (MW)", precision=1)
                kpi_ep_reward  = gr.Number(label="Episode Reward",     precision=3)

            result_md = gr.Markdown("_Run an episode to see results._")

        # -- Tab 2: Grid Diagram -------------------------------------------
        with gr.Tab("Grid Diagram"):
            grid_plot    = gr.Plot(label="IEEE 14-Bus Network")
            reward_plot  = gr.Plot(label="Reward Breakdown")
            history_plot = gr.Plot(label="Reward per Step")

        # -- Tab 3: Raw State ----------------------------------------------
        with gr.Tab("Raw State"):
            state_json = gr.JSON(label="Current Environment State")

    # -- Callbacks ---------------------------------------------------------

    def do_reset(diff):
        global _env, _agent, _step_rewards, _last_state
        with _lock:
            _env          = PowerGridEnv(Difficulty(diff), seed=None)
            state         = _env.reset()
            _last_state   = state
            _step_rewards = []
            _agent        = None
        fig_g = _make_grid_figure(state)
        rc    = {k: 0.0 for k in ("balance","overload","reserve","renewable","cost","stability")}
        fig_r = _make_reward_bar(rc, diff)
        return (state, fig_g, fig_r, None,
                0, 0, 0, 0, 0, 0,
                state["total_gen_mw"], state["total_load_mw"],
                state["power_balance_mw"], 0.0,
                "_Environment reset. Press **Agent Step** or **Run Episode**._")

    def do_agent_step(agent_name):
        global _agent
        if _env is None:
            return [None] * 14 + ["Reset first."]
        with _lock:
            if _agent is None:
                _agent = AGENT_REGISTRY.get(agent_name, EconomicDispatchAgent)()
            state  = _env.state()
            action = _agent.act(state, _env)
            try:
                state, reward, done, info = _env.step(action)
            except Exception as e:
                return [gr.skip()] * 14 + ["**Error**: " + str(e)]
            _last_state.update(state)
            _step_rewards.append(reward)
        fig_g    = _make_grid_figure(state)
        fig_r    = _make_reward_bar(info["reward_components"], _env.difficulty.value)
        fig_hist = _make_reward_history_plot(_step_rewards)
        rc   = info["reward_components"]
        ep_r = _env.episode_reward
        msg  = ("Step " + str(info["step"]) + " | Reward: **" + str(round(reward, 3)) + "**"
                + (" | Relay trip!" if info["relay_tripped"] else "")
                + (" | Disturbance!" if info["disturbance"]["type"] else "")
                + (" | DONE." if done else ""))
        return (state, fig_g, fig_r, fig_hist,
                rc["balance"], rc["overload"], rc["reserve"],
                rc["renewable"], rc["cost"], rc["stability"],
                state["total_gen_mw"], state["total_load_mw"],
                state["power_balance_mw"], ep_r, msg)

    def do_run_episode(diff, agent_name, n_eps):
        global _env, _agent, _step_rewards, _last_state
        results    = []
        all_rw: list = []

        for ep in range(int(n_eps)):
            with _lock:
                _env          = PowerGridEnv(Difficulty(diff), seed=ep)
                state         = _env.reset()
                _step_rewards = []
                ag = AGENT_REGISTRY.get(agent_name, EconomicDispatchAgent)()
            done = False
            while not done:
                with _lock:
                    action = ag.act(state, _env)
                    state, r, done, info = _env.step(action)
                    _step_rewards.append(r)
                    all_rw.append(r)
            with _lock:
                result = grade_episode(_env)
                result["step_rewards"] = _step_rewards[:]
                results.append(result)
                _last_state = state

        fig_g    = _make_grid_figure(state)
        rc       = info["reward_components"]
        fig_r    = _make_reward_bar(rc, diff)
        fig_hist = _make_reward_history_plot(all_rw)

        n_passed = sum(1 for r in results if r["passed"])
        lines = [
            "### Results: " + str(int(n_eps)) + " episode(s) on **" + diff.upper() + "** "
            "| Passed: **" + str(n_passed) + "/" + str(int(n_eps)) + "**\n"
        ]
        for ep, r in enumerate(results, 1):
            m = r["metrics"]
            lines.append(
                "**Ep " + str(ep) + "** | Grade: **" + r["grade"] + "** "
                "| Score: " + str(r["score_01"]) + " "
                "| Avg R: " + str(round(m["avg_reward/step"], 3)) + " "
                "| Overloads: " + str(m["overload_events"]) + " "
                "| Relays: " + str(m["relay_trips"]) + " "
                "| Passed: **" + ("YES" if r["passed"] else "NO") + "**"
            )
        ep_r = _env.episode_reward
        return (state, fig_g, fig_r, fig_hist,
                rc["balance"], rc["overload"], rc["reserve"],
                rc["renewable"], rc["cost"], rc["stability"],
                state["total_gen_mw"], state["total_load_mw"],
                state["power_balance_mw"], ep_r,
                "\n\n".join(lines))

    OUTPUTS = [
        state_json, grid_plot, reward_plot, history_plot,
        kpi_balance, kpi_overload, kpi_reserve,
        kpi_renewable, kpi_cost, kpi_stability,
        kpi_total_gen, kpi_total_load, kpi_balance_mw, kpi_ep_reward,
        result_md,
    ]

    btn_reset.click(do_reset,        inputs=[diff_dd],                     outputs=OUTPUTS)
    btn_step.click( do_agent_step,   inputs=[agent_dd],                    outputs=OUTPUTS)
    btn_run.click(  do_run_episode,  inputs=[diff_dd, agent_dd, episodes], outputs=OUTPUTS)


# =============================================================================
# Mount Gradio onto FastAPI and launch
# =============================================================================

app = gr.mount_gradio_app(fastapi_app, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print("Power Grid Optimizer -> http://{}:{}".format(display_host, args.port))
    print("  Gradio UI  -> /")
    print("  REST API   -> /api/  (try /api/info)")
    print("  API docs   -> /docs\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
