"""
client.py
---------
PowerGridEnv — OpenEnv async + sync client for the Power Grid environment.

Mirrors the exact structure of carla_env/client.py, calendar_env/client.py,
and reasoning_gym_env/client.py for maximum compatibility.

Quick Start
-----------
    # Async (default)
    from power_grid_env import PowerGridEnv, PowerGridAction

    async with PowerGridEnv(base_url="http://localhost:7860") as env:
        result = await env.reset(difficulty="hard", scenario_id="cascade_blackout")
        print(result.observation.scenario_description)
        result = await env.step(PowerGridAction(dispatch_mw=[100, 45, 45, 45, 40, 28]))
        print(f"Reward: {result.reward}")

    # Sync wrapper
    with PowerGridEnv(base_url="http://localhost:7860").sync() as env:
        result = env.reset(difficulty="hard", scenario_id="cascade_blackout")
        result = env.step(PowerGridAction(dispatch_mw=[100, 45, 45, 45, 40, 28]))

    # From Docker image (auto-starts container)
    env = PowerGridEnv.from_docker_image("power-grid:latest")

No local setup needed — point your client at the live Hugging Face Space:
    async with PowerGridEnv(base_url="https://<your-hf-space>.hf.space") as env:
        ...
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from contextlib import asynccontextmanager, contextmanager
from typing import List, Optional

import httpx

from models import (
    PowerGridAction,
    PowerGridObservation,
    PowerGridState,
)


# ─────────────────────────────────────────────────────────────────────────────
# Step result wrapper
# ─────────────────────────────────────────────────────────────────────────────

class StepResult:
    """Unified result returned by both reset() and step()."""
    def __init__(self, data: dict, is_reset: bool = False):
        raw_state = data.get("state", data)
        self.observation    = PowerGridObservation(**raw_state)
        self.reward: float  = data.get("reward", 0.0)
        self.done:   bool   = data.get("done",   False)
        self.info:   dict   = data.get("info",   {})
        self.is_reset       = is_reset

    def __repr__(self):
        return (
            f"StepResult(step={self.observation.step}, "
            f"reward={self.reward:.3f}, done={self.done}, "
            f"balance={self.observation.power_balance_mw:+.1f} MW)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Async client
# ─────────────────────────────────────────────────────────────────────────────

class PowerGridEnv:
    """Async OpenEnv client for the Power Grid environment.

    Parameters
    ----------
    base_url : str
        URL of the running environment server.
    timeout  : float
        HTTP request timeout in seconds.
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._docker_container_id: Optional[str] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "PowerGridEnv":
        self._client = httpx.AsyncClient(base_url=self._base, timeout=self._timeout)
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._docker_container_id:
            subprocess.run(["docker", "stop", self._docker_container_id],
                           capture_output=True)
            self._docker_container_id = None

    # ── Factory method: from Docker image ─────────────────────────────────────

    @classmethod
    def from_docker_image(
        cls,
        image: str = "power-grid:latest",
        port: int = 7860,
        timeout: float = 60.0,
    ) -> "PowerGridEnv":
        """Start a Docker container and connect to it.

        Handles: starting container, waiting for readiness, cleanup on close().
        """
        proc = subprocess.run(
            ["docker", "run", "-d", "-p", f"{port}:{port}", image],
            capture_output=True, text=True
        )
        container_id = proc.stdout.strip()
        env = cls(base_url=f"http://localhost:{port}", timeout=timeout)
        env._docker_container_id = container_id

        # Wait for server readiness
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                httpx.get(f"http://localhost:{port}/health", timeout=2.0)
                break
            except Exception:
                time.sleep(1.0)
        return env

    # ── Sync wrapper ──────────────────────────────────────────────────────────

    def sync(self) -> "_SyncWrapper":
        """Return a synchronous wrapper for non-async usage."""
        return _SyncWrapper(self)

    # ── Core API ──────────────────────────────────────────────────────────────

    async def reset(
        self,
        difficulty: str = "easy",
        scenario_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> StepResult:
        """Reset the environment and return the initial observation.

        Parameters
        ----------
        difficulty  : "easy" | "medium" | "hard"
        scenario_id : optional named scenario (e.g. "cascade_blackout")
        seed        : optional random seed for reproducibility
        """
        params: dict = {"difficulty": difficulty}
        if scenario_id:
            params["scenario_id"] = scenario_id
        if seed is not None:
            params["seed"] = seed

        resp = await self._client.post("/api/reset", params=params)
        resp.raise_for_status()
        return StepResult(resp.json(), is_reset=True)

    async def step(self, action: PowerGridAction) -> StepResult:
        """Advance one step with the given generator dispatch action.

        Parameters
        ----------
        action : PowerGridAction
            dispatch_mw: list of 6 MW setpoints for generators 0-5
        """
        resp = await self._client.post(
            "/api/step",
            json={"action": action.dispatch_mw},
        )
        resp.raise_for_status()
        return StepResult(resp.json())

    async def state(self) -> PowerGridState:
        """Get the current server-side state without advancing."""
        resp = await self._client.get("/api/state")
        resp.raise_for_status()
        data = resp.json()
        obs = PowerGridObservation(**data)
        return PowerGridState(observation=obs)

    async def info(self) -> dict:
        """Return static environment topology and generator metadata."""
        resp = await self._client.get("/api/info")
        resp.raise_for_status()
        return resp.json()

    async def scenarios(self, probe_only: bool = False, trainable_only: bool = False) -> List[dict]:
        """List available named crisis scenarios."""
        params = {}
        if probe_only:
            params["probe_only"] = "true"
        if trainable_only:
            params["trainable_only"] = "true"
        resp = await self._client.get("/api/scenarios", params=params)
        resp.raise_for_status()
        return resp.json()["scenarios"]

    async def health(self) -> dict:
        """Check server health."""
        resp = await self._client.get("/health")
        resp.raise_for_status()
        return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# Synchronous wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _SyncWrapper:
    """Synchronous wrapper around the async PowerGridEnv client."""

    def __init__(self, env: PowerGridEnv):
        self._env = env
        self._loop = asyncio.new_event_loop()

    def __enter__(self) -> "_SyncWrapper":
        self._loop.run_until_complete(self._env.__aenter__())
        return self

    def __exit__(self, *args) -> None:
        self._loop.run_until_complete(self._env.__aexit__(*args))
        self._loop.close()

    def reset(self, difficulty: str = "easy",
              scenario_id: Optional[str] = None,
              seed: Optional[int] = None) -> StepResult:
        return self._loop.run_until_complete(
            self._env.reset(difficulty=difficulty, scenario_id=scenario_id, seed=seed)
        )

    def step(self, action: PowerGridAction) -> StepResult:
        return self._loop.run_until_complete(self._env.step(action))

    def state(self) -> PowerGridState:
        return self._loop.run_until_complete(self._env.state())

    def info(self) -> dict:
        return self._loop.run_until_complete(self._env.info())

    def scenarios(self, **kwargs) -> List[dict]:
        return self._loop.run_until_complete(self._env.scenarios(**kwargs))


# ─────────────────────────────────────────────────────────────────────────────
# Module exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = ["PowerGridEnv", "PowerGridAction", "PowerGridObservation", "PowerGridState"]


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo (run as script)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    scenario = sys.argv[2] if len(sys.argv) > 2 else "cascade_blackout"

    async def demo():
        async with PowerGridEnv(base_url=base) as env:
            print(f"\n⚡ Power Grid Environment — {base}")
            print(f"   Scenario: {scenario}\n")

            result = await env.reset(difficulty="hard", scenario_id=scenario)
            obs = result.observation
            print(f"Scenario: {obs.scenario_id}")
            print(f"Description: {obs.scenario_description}")
            print(f"Total Load: {obs.total_load_mw:.1f} MW")
            print(f"Available Gen: {obs.gen_available_mw}")
            print(f"Generators Online: {obs.gen_online}\n")

            action = PowerGridAction(dispatch_mw=[100.0, 0.0, 50.0, 45.0, 40.0, 28.0])
            result = await env.step(action)
            print(f"Step 1 → Reward: {result.reward:+.3f}  Done: {result.done}")
            print(f"Balance: {result.observation.power_balance_mw:+.1f} MW")
            print(f"Relay tripped: {result.observation.relay_tripped}")

            scenarios = await env.scenarios()
            print(f"\nAvailable scenarios ({len(scenarios)}):")
            for s in scenarios:
                tag = "🔍 PROBE" if s["probe"] else "🏋️ TRAIN"
                print(f"  {tag}  {s['id']:30s}  {s['name']}")

    asyncio.run(demo())
