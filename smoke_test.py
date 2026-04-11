import sys, os
sys.path.insert(0, '.')
from scenarios import list_scenarios, get_scenario, SCENARIO_REGISTRY

print(f'Scenarios loaded: {len(SCENARIO_REGISTRY)}')
for s in SCENARIO_REGISTRY.values():
    tag = 'PROBE' if s.probe else 'TRAIN'
    print(f'  [{tag}] {s.id}: {s.name}')

# Test inject
from env.grid_env import PowerGridEnv, Difficulty
env = PowerGridEnv(Difficulty.HARD, seed=101)
env.reset()
sc = get_scenario('cascade_blackout')
sc.inject(env)
state = env.state()
print('\ncascade_blackout injected:')
print(f"  Gen online: {state['gen_online']}")
print(f"  Total load: {state['total_load_mw']:.1f} MW")
print(f"  Total gen:  {state['total_gen_mw']:.1f} MW")
print(f"  Balance:    {state['power_balance_mw']:+.1f} MW")

print()
from models import PowerGridAction, PowerGridObservation
a = PowerGridAction(dispatch_mw=[80, 60, 40, 45, 35, 25])
print(f'Action: {a.dispatch_mw}')

# Test all 8 scenarios
print('\nAll scenario inject tests:')
for sid, sc in SCENARIO_REGISTRY.items():
    env2 = PowerGridEnv(Difficulty(sc.difficulty), seed=sc.seed)
    env2.reset()
    sc.inject(env2)
    s2 = env2.state()
    print(f'  {sid}: gen_online={s2["gen_online"]} load={s2["total_load_mw"]:.1f} MW')

print('\nAll OK!')
