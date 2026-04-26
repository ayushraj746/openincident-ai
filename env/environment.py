from typing import Tuple
import random

from .state import State
from .tasks import generate_task
from reward.reward import RewardEngine


class OpenIncidentEnv:
    def __init__(self, difficulty: str = "easy", eval_mode: bool = False):
        self.difficulty = difficulty
        self.eval_mode = eval_mode

        self.state: State = None
        self.done = False

        self.steps = 0
        self.max_steps = 20

        self.reward_engine = RewardEngine()

    # ---------------- SAFE VALUE ---------------- #

    def _safe(self, val, default):
        return val if val is not None else default

    # ---------------- OBSERVATION ---------------- #

    def _get_observation(self):

        # base values
        cpu = self._safe(self.state.cpu_usage, 50)
        latency = self._safe(self.state.latency, 500)
        memory = self._safe(self.state.memory_usage, 50)

        # add noise
        cpu += random.uniform(-3, 3)
        latency += random.uniform(-50, 50)
        memory += random.uniform(-5, 5)

        network = self._safe(self.state.network_status, "normal")

        # 🔥 PARTIAL OBSERVABILITY (SAFE VERSION)
        if random.random() < 0.1:
            cpu = 50
        if random.random() < 0.1:
            latency = 500
        if random.random() < 0.1:
            memory = 50

        obs = {
            "cpu_usage": cpu,
            "latency": latency,
            "service_health": self._safe(self.state.service_health, "degraded"),
            "memory_usage": memory,
            "network_status": network,
            "severity": getattr(self.state, "severity", "low"),
            "available_agents": getattr(self.state, "available_agents", []),
        }

        # ---------------- SCHEMA DRIFT ---------------- #

        if random.random() < 0.2:
            obs = {
                "metrics": {
                    "processor_load": obs["cpu_usage"],
                    "response_time": obs["latency"],
                },
                "system": {
                    "memory": obs["memory_usage"],
                    "network": obs["network_status"],
                    "service": obs["service_health"],
                },
                "context": {
                    "severity": obs["severity"],
                    "agents": obs["available_agents"],
                },
            }

        return obs

    # ---------------- RESET ---------------- #

    def reset(self) -> dict:
        if self.eval_mode:
            self.state = State(
                cpu_usage=90,
                latency=1200,
                memory_usage=90,
                network_status="down",
                service_health="down",
            )
        else:
            self.state = generate_task(self.difficulty)

        self.done = False
        self.steps = 0
        self.state.step_count = 0

        return self._get_observation()

    # ---------------- STEP ---------------- #

    def step(self, action: str) -> Tuple[dict, float, bool, dict]:

        self.steps += 1
        self.state.step_count += 1

        prev_state = self.state.to_dict()

        # ---------------- ACTION EFFECTS ---------------- #

        if action == "delegate_sre":
            self.state.cpu_usage *= 0.7
            self.state.latency *= 0.75

        elif action == "delegate_memory":
            self.state.memory_usage *= 0.7

        elif action in ["delegate_network", "restart_network"]:
            if self.state.network_status in ["down", None]:
                self.state.network_status = "slow"
            elif self.state.network_status == "slow":
                self.state.network_status = "normal"
            else:
                self.state.network_status = "normal"

        elif action == "rollback_deployment":
            self.state.cpu_usage *= 0.6
            self.state.memory_usage *= 0.6
            self.state.latency *= 0.7

        if action == "restart_service":
            if self.state.network_status == "normal":
                self.state.service_health = "healthy"
            elif self.state.network_status == "slow":
                self.state.service_health = "degraded"

        # ---------------- NATURAL RECOVERY ---------------- #

        if self.state.latency > 400:
            self.state.latency -= random.randint(50, 120)

        if self.state.cpu_usage > 40:
            self.state.cpu_usage -= random.randint(3, 10)

        if self.state.memory_usage > 40:
            self.state.memory_usage -= random.randint(5, 12)

        # ---------------- RANDOM SPIKES ---------------- #

        if random.random() < 0.15:
            self.state.cpu_usage += random.randint(5, 10)

        if random.random() < 0.1:
            self.state.latency += random.randint(80, 200)

        if random.random() < 0.05:
            if self.state.network_status == "normal":
                self.state.network_status = "slow"

        # ---------------- SAFETY ---------------- #

        self.state.cpu_usage = max(0, min(100, self.state.cpu_usage))
        self.state.latency = max(0, self.state.latency)
        self.state.memory_usage = max(0, min(100, self.state.memory_usage))

        # ---------------- AUTO RECOVERY ---------------- #

        if (
            self.state.cpu_usage < 60
            and self.state.memory_usage < 60
            and self.state.network_status == "normal"
        ):
            self.state.service_health = "healthy"

        # ---------------- TERMINATION ---------------- #

        if (
            self.state.cpu_usage < 65
            and self.state.latency <= 600
            and self.state.memory_usage < 65
            and self.state.network_status == "normal"
            and self.state.service_health == "healthy"
        ):
            self.done = True

        if self.steps >= self.max_steps:
            self.done = True

        # ---------------- REWARD ---------------- #

        current_state = self.state.to_dict()

        reward = self.reward_engine.compute_reward(
            prev_state=prev_state,
            current_state=current_state,
            action=action,
            done=self.done,
        )

        return self._get_observation(), reward, self.done, {}