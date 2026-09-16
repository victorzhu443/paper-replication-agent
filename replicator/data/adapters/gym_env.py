"""RL environments via gymnasium / ale-py. Probe = the env id can be constructed. No data rows."""
from __future__ import annotations

import importlib.util

import pandas as pd

KEYWORDS = ("atari", "gym", "cartpole", "mujoco", "ale", "lunarlander", "pong", "breakout", "carracing", "vizdoom", "doom",
            "environment", "env")


def matches(canonical: str) -> bool:
    c = canonical.lower()
    return any(k in c for k in KEYWORDS)


def probe(env_id: str = "CartPole-v1") -> tuple[bool, str]:
    if importlib.util.find_spec("gymnasium") is None:
        return False, "gymnasium not installed"
    try:
        import gymnasium as gym
        if "ALE/" in env_id or "atari" in env_id.lower():
            import ale_py  # noqa: F401
            gym.register_envs(ale_py)
        env = gym.make(env_id)
        env.close()
        return True, f"gymnasium env {env_id} constructed"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:120]}"


def fetch(params: dict) -> tuple[pd.DataFrame, str]:
    return pd.DataFrame([{"env_id": params.get("env_id", "CartPole-v1")}]), f"gym:{params.get('env_id', 'CartPole-v1')}"
