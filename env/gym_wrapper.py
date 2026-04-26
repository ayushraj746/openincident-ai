import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .environment import OpenIncidentEnv


class GymOpenIncidentEnv(gym.Env):
    def __init__(self, difficulty="medium", eval_mode=False):
        super().__init__()

        self.env = OpenIncidentEnv(difficulty=difficulty, eval_mode=eval_mode)

        # ✅ MATCH TRAINING (7 FEATURES ONLY)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

        # ACTION SPACE
        self.action_map = {
            0: "delegate_sre",
            1: "delegate_network",
            2: "delegate_memory",
            3: "rollback_deployment",
            4: "restart_service",
            5: "do_nothing",
        }

        self.action_space = spaces.Discrete(len(self.action_map))

    def reset(self, seed=None, options=None):
        state = self.env.reset()
        return self._convert_state(state), {}

    def step(self, action):
        action_name = self.action_map[int(action)]

        state, reward, done, info = self.env.step(action_name)

        return self._convert_state(state), reward, done, False, info

    def _safe(self, value, default):
        return value if value is not None else default

    def _convert_state(self, state):

        # ---------------- HANDLE SCHEMA ---------------- #

        if "metrics" in state:
            cpu = self._safe(state["metrics"].get("processor_load"), 50)
            latency = self._safe(state["metrics"].get("response_time"), 500)
            memory = self._safe(state["system"].get("memory"), 50)

            network_status = state["system"].get("network", "normal")
            service_health = state["system"].get("service", "degraded")

            context = state.get("context", {})
            severity_val = context.get("severity", "low")

        else:
            cpu = self._safe(state.get("cpu_usage"), 50)
            latency = self._safe(state.get("latency"), 500)
            memory = self._safe(state.get("memory_usage"), 50)

            network_status = state.get("network_status", "normal")
            service_health = state.get("service_health", "degraded")

            severity_val = state.get("severity", "low")

        # ---------------- ENCODING ---------------- #

        severity_map = {"low": 0, "medium": 1, "high": 2}
        network_map = {"normal": 0, "slow": 1, "down": 2}
        service_map = {"healthy": 2, "degraded": 1, "down": 0}

        severity = severity_map.get(severity_val, 0)
        network = network_map.get(network_status, 0)
        service = service_map.get(service_health, 1)

        # ---------------- NORMALIZATION ---------------- #

        cpu = cpu / 100.0
        latency = latency / 2000.0
        memory = memory / 100.0

        network = network / 2.0
        service = service / 2.0
        severity = severity / 2.0

        # ---------------- FINAL VECTOR (7 FEATURES) ---------------- #

        return np.array(
            [
                cpu,
                latency,
                memory,
                network,
                service,
                severity,
                cpu * latency,  # interaction (keeps richness)
            ],
            dtype=np.float32,
        )