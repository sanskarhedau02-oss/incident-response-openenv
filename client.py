"""
client.py — Python client for the AIOps OpenEnv REST server
"""

import requests


class AIOpsEnv:
    """
    Thin HTTP wrapper around the AIOps environment server.

    Example
    -------
        env = AIOpsEnv()
        obs = env.reset(task="hard")
        while True:
            action = {"action_type": "scale_up", "target": None}
            result = env.step(action)
            if result["done"]:
                break
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def reset(self, task: str = "easy") -> dict:
        return requests.post(f"{self.base_url}/reset", params={"task": task}).json()

    def step(self, action: dict) -> dict:
        return requests.post(f"{self.base_url}/step", json=action).json()

    def state(self) -> dict:
        return requests.get(f"{self.base_url}/state").json()

    def observation_vector(self) -> list[float]:
        return requests.get(f"{self.base_url}/observation").json()["observation_vector"]

    def info(self) -> dict:
        return requests.get(f"{self.base_url}/info").json()

    def health(self) -> dict:
        return requests.get(f"{self.base_url}/health").json()
