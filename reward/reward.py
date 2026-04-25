from typing import Dict


class RewardEngine:
    def __init__(self):
        self.last_action = None

    def compute_reward(
        self,
        prev_state: Dict,
        current_state: Dict,
        action: str,
        done: bool,
    ) -> float:

        reward = 0.0

        # ---------------- CONTEXT ---------------- #

        severity = current_state.get("severity", "low")

        severity_weight = {
            "low": 1.0,
            "medium": 1.5,
            "high": 2.0,
        }

        weight = severity_weight.get(severity, 1.0)

        # ---------------- STATE VALUES ---------------- #

        prev_cpu = prev_state.get("cpu_usage", 50)
        curr_cpu = current_state.get("cpu_usage", 50)

        prev_latency = prev_state.get("latency", 500)
        curr_latency = current_state.get("latency", 500)

        prev_memory = prev_state.get("memory_usage", 50)
        curr_memory = current_state.get("memory_usage", 50)

        prev_network = prev_state.get("network_status", "normal")
        curr_network = current_state.get("network_status", "normal")

        prev_service = prev_state.get("service_health", "degraded")
        curr_service = current_state.get("service_health", "degraded")

        # ---------------- POSITIVE SIGNALS ---------------- #

        if curr_cpu < prev_cpu:
            reward += 0.3 * weight

        if curr_latency < prev_latency:
            reward += 0.3 * weight

        if curr_memory < prev_memory:
            reward += 0.2 * weight

        # network improvement
        if prev_network == "down" and curr_network in ["slow", "normal"]:
            reward += 0.6 * weight
        elif prev_network == "slow" and curr_network == "normal":
            reward += 0.4 * weight

        # service recovery
        if prev_service != "healthy" and curr_service == "healthy":
            reward += 0.8 * weight

        # ---------------- INTELLIGENT ACTION REWARD ---------------- #

        if prev_cpu > 80 and action == "delegate_sre":
            reward += 0.5

        if prev_latency > 800 and action == "delegate_sre":
            reward += 0.5

        if prev_network == "down" and action in ["delegate_network", "restart_network"]:
            reward += 0.5

        if prev_memory > 80 and action == "delegate_memory":
            reward += 0.4

        if prev_service == "down" and action == "restart_service":
            reward += 0.4

        # ---------------- WRONG ACTION PENALTIES ---------------- #

        if prev_cpu > 80 and action not in ["delegate_sre"]:
            reward -= 0.3

        if prev_network == "down" and action not in ["delegate_network", "restart_network"]:
            reward -= 0.3

        if prev_memory > 80 and action not in ["delegate_memory"]:
            reward -= 0.2

        # ---------------- NEGATIVE SIGNALS ---------------- #

        # repetition penalty
        if self.last_action == action:
            reward -= 0.3

        # do nothing penalty (stronger)
        if action == "do_nothing" and not done:
            reward -= 0.3

        # no improvement penalty
        if (
            curr_cpu >= prev_cpu
            and curr_latency >= prev_latency
            and curr_memory >= prev_memory
        ):
            reward -= 0.3

        # oscillation penalty
        if self.last_action and self.last_action != action:
            reward -= 0.05

        # ---------------- EFFICIENCY BONUS ---------------- #

        # encourage faster stabilization
        if (
            curr_cpu < 60
            and curr_latency < 600
            and curr_memory < 60
        ):
            reward += 0.2

        # ---------------- TERMINAL BONUS ---------------- #

        if done:
            if severity == "high":
                reward += 6.0
            elif severity == "medium":
                reward += 4.0
            else:
                reward += 3.0

        # ---------------- UPDATE MEMORY ---------------- #

        self.last_action = action

        return reward