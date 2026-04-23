from typing import Dict


class RewardEngine:
    def __init__(self):
        self.last_action = None  # 🔥 track previous action

    def compute_reward(
        self,
        prev_state: Dict,
        current_state: Dict,
        action: str,
        done: bool,
    ) -> float:
        reward = 0.0

        # ---------------- POSITIVE SIGNALS ---------------- #

        # CPU improvement
        if current_state["cpu_usage"] < prev_state["cpu_usage"]:
            reward += 0.3

        # Latency improvement
        if current_state["latency"] < prev_state["latency"]:
            reward += 0.3

        # Memory improvement (NEW 🔥)
        if current_state.get("memory_usage", 0) < prev_state.get("memory_usage", 0):
            reward += 0.2

        # Service recovery
        if (
            prev_state["service_health"] != "healthy"
            and current_state["service_health"] == "healthy"
        ):
            reward += 0.5

        # Network recovery (NEW 🔥)
        if prev_state.get("network_status") == "down" and current_state.get("network_status") in ["slow", "normal"]:
            reward += 0.4

        # ---------------- NEGATIVE SIGNALS ---------------- #

        # 🔴 Penalize repeating same action
        if self.last_action == action:
            reward -= 0.3

        # discourage useless action
        if action == "do_nothing":
            reward -= 0.1

        # mild penalty if nothing improved
        if (
            current_state["cpu_usage"] >= prev_state["cpu_usage"]
            and current_state["latency"] >= prev_state["latency"]
        ):
            reward -= 0.1

        # ---------------- TERMINAL BONUS ---------------- #

        if done:
            reward += 2.0

        # ---------------- UPDATE MEMORY ---------------- #

        self.last_action = action

        return reward