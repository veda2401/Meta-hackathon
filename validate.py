"""
validate.py
-----------
Quick environment contract validation — 13 checks, CI-ready.
Exit code 0 = all pass; exit code 1 = failures found.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

CHECK = "✅"; FAIL = "❌"
failures = []

def check(name, fn):
    try:
        fn(); print(f"  {CHECK}  {name}")
    except Exception as e:
        failures.append(name); print(f"  {FAIL}  {name}\n       → {e}")

# ── imports ────────────────────────────────────────────────────────────────
from env.grid_env import PowerGridEnv, Difficulty, PROFILES
from env.dc_power_flow import build_b_matrix, solve_dc_power_flow
from env import ieee14
from agents.baselines import EconomicDispatchAgent
from tasks.graders import grade_episode

def _c_b_matrix_shape():
    B = build_b_matrix([True]*ieee14.N_LINES)
    assert B.shape == (ieee14.N_BUSES, ieee14.N_BUSES)

def _c_b_matrix_symmetric():
    B = build_b_matrix([True]*ieee14.N_LINES)
    np.testing.assert_allclose(B, B.T, atol=1e-10)

def _c_b_matrix_row_sums_zero():
    B = build_b_matrix([True]*ieee14.N_LINES)
    np.testing.assert_allclose(B.sum(axis=1), 0, atol=1e-8)

def _c_dc_converges():
    env = PowerGridEnv(Difficulty.EASY, seed=0)
    s   = env.reset()
    p   = np.zeros(ieee14.N_BUSES)
    for i in range(ieee14.N_GENS):
        p[ieee14.GEN_BUS_IDX[i]] += s["gen_dispatch_mw"][i]
    p -= np.array(s["bus_loads_mw"])
    r = solve_dc_power_flow(p, [True]*ieee14.N_LINES)
    assert r.converged, "DC solver must converge on full network"

def _c_slack_angle_zero():
    env = PowerGridEnv(Difficulty.EASY, seed=0)
    s   = env.reset()
    assert abs(s["bus_angles_deg"][ieee14.SLACK_IDX]) < 1e-6

def _c_reset_returns_dict():
    for d in Difficulty:
        s = PowerGridEnv(d, seed=0).reset()
        assert isinstance(s, dict)

def _c_step_tuple():
    env = PowerGridEnv(Difficulty.EASY, seed=0)
    env.reset()
    res = env.step(np.array([50.0]*ieee14.N_GENS))
    assert len(res) == 4

def _c_reward_in_range():
    env  = PowerGridEnv(Difficulty.HARD, seed=0)
    s    = env.reset()
    for _ in range(30):
        a = np.array([min(ieee14.GEN_P_MAX[i], s["gen_available_mw"][i])*0.6
                      for i in range(ieee14.N_GENS)])
        s, r, done, info = env.step(a)
        assert -1.0-1e-6 <= r <= 1.0+1e-6, f"reward {r} out of [-1,1]"
        if done: break

def _c_easy_no_disturbances():
    assert len(PROFILES[Difficulty.EASY].disturbance_types) == 0

def _c_hard_all_disturbances():
    from env.grid_env import DisturbanceType
    assert all(dt in PROFILES[Difficulty.HARD].disturbance_types for dt in DisturbanceType)

def _c_tripped_line_zero_flow():
    env = PowerGridEnv(Difficulty.HARD, seed=0)
    env.reset()
    env._line_status[0] = False
    env._relay_restore[0] = 9999
    env._last_dc = env._run_power_flow()
    assert abs(env.state()["line_flows_mw"][0]) < 1e-5

def _c_episode_terminates():
    env  = PowerGridEnv(Difficulty.EASY, seed=0)
    ag   = EconomicDispatchAgent()
    s    = env.reset()
    done = False
    steps = 0
    while not done:
        s, _, done, _ = env.step(ag.act(s, env))
        steps += 1
    assert steps == env.max_steps

def _c_grader_valid():
    env = PowerGridEnv(Difficulty.MEDIUM, seed=0)
    ag  = EconomicDispatchAgent()
    s   = env.reset()
    done = False
    while not done:
        s, _, done, _ = env.step(ag.act(s, env))
    r = grade_episode(env)
    assert r["grade"] in "ABCDF"
    assert 0.0 < r["score"] < 1.0,    f"score {r['score']} not in (0,1)"
    assert 0.0 < r["score_01"] < 1.0, f"score_01 {r['score_01']} not in (0,1)"

# ─────────────────────────────────────────
print("\n🔌 Power Grid Environment — Validation Suite\n" + "─"*50)

check("B-matrix has correct shape",            _c_b_matrix_shape)
check("B-matrix is symmetric",                 _c_b_matrix_symmetric)
check("B-matrix row sums equal zero (KCL)",    _c_b_matrix_row_sums_zero)
check("DC power flow converges (full network)",_c_dc_converges)
check("Slack bus angle = 0",                   _c_slack_angle_zero)
check("reset() returns dict (3 difficulties)", _c_reset_returns_dict)
check("step() returns 4-tuple",                _c_step_tuple)
check("reward always in [-1, +1]",             _c_reward_in_range)
check("EASY has zero disturbance types",       _c_easy_no_disturbances)
check("HARD has all 3 disturbance types",      _c_hard_all_disturbances)
check("tripped line carries 0 MW",             _c_tripped_line_zero_flow)
check("episode terminates at max_steps",       _c_episode_terminates)
check("grader returns valid score + grade",    _c_grader_valid)

print("─"*50)
if failures:
    print(f"\n{FAIL} {len(failures)} check(s) FAILED: {failures}")
    sys.exit(1)
else:
    print(f"\n{CHECK} All 13 checks passed!\n")
    sys.exit(0)
