import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .environment import OpenIncidentEnv


class GymOpenIncidentEnv(gym.Env):
    def __init__(self, difficulty="medium", eval_mode=False):
        super().__init__()

        self.env = OpenIncidentEnv(difficulty=difficulty, eval_mode=eval_mode)

        # Observation space (7 features)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

        # Action space (5 actions)
        self.action_space = spaces.Discrete(5)

    def reset(self, seed=None, options=None):
        state = self.env.reset()
        return self._convert_state(state), {}

    def step(self, action):

        action_map = {
            0: "delegate_sre",
            1: "delegate_network",
            2: "delegate_memory",
            3: "rollback_deployment",
            4: "do_nothing",
        }

        state, reward, done, _ = self.env.step(action_map[action])

        return self._convert_state(state), reward, done, False, {}

    def _safe_value(self, value, default):
        """Handles None values safely"""
        return value if value is not None else default

    def _convert_state(self, state):

        # ---------------- SCHEMA DRIFT HANDLING ---------------- #

        if "metrics" in state:
            cpu = self._safe_value(state["metrics"].get("processor_load"), 50)
            latency = self._safe_value(state["metrics"].get("response_time"), 500)
            memory = self._safe_value(state["system"].get("memory"), 50)

            network_status = state["system"].get("network", "normal")
            service_health = state["system"].get("service", "degraded")

            # context may be nested
            context = state.get("context", {})
            severity_val = context.get("severity", "low")
            agents = context.get("agents", [])

        else:
            cpu = self._safe_value(state.get("cpu_usage"), 50)
            latency = self._safe_value(state.get("latency"), 500)
            memory = self._safe_value(state.get("memory_usage"), 50)

            network_status = state.get("network_status", "normal")
            service_health = state.get("service_health", "degraded")

            severity_val = state.get("severity", "low")
            agents = state.get("available_agents", [])

        # ---------------- NEW FEATURES ---------------- #

        severity_map = {"low": 0, "medium": 1, "high": 2}
        severity = severity_map.get(severity_val, 0)

        agent_count = len(agents)

        # ---------------- CATEGORICAL ENCODING ---------------- #

        network_map = {"normal": 0, "slow": 1, "down": 2}
        service_map = {"healthy": 2, "degraded": 1, "down": 0}

        network = network_map.get(network_status, 0)
        service = service_map.get(service_health, 1)

        # ---------------- NORMALIZATION ---------------- #

        cpu = cpu / 100.0
        latency = latency / 2000.0
        memory = memory / 100.0
        network = network / 2.0
        service = service / 2.0
        severity = severity / 2.0
        agent_count = agent_count / 3.0

        return np.array(
            [cpu, latency, memory, network, service, severity, agent_count],
            dtype=np.float32,
        )