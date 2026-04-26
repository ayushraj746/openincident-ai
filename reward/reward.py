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
        info: Dict = None,   # 🔥 NEW (agent feedback)
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

        # ---------------- VERIFIABLE OUTCOME (CORE) ---------------- #

        system_improved = (
            curr_cpu < prev_cpu
            or curr_latency < prev_latency
            or curr_memory < prev_memory
        )

        if system_improved:
            reward += 0.5 * weight

        # strict success condition (hard check)
        system_healthy = (
            curr_cpu < 60
            and curr_latency < 600
            and curr_memory < 60
            and curr_network == "normal"
            and curr_service == "healthy"
        )

        if system_healthy:
            reward += 2.0 * weight

        # ---------------- AGENT FEEDBACK (NEW) ---------------- #

        if info:
            agent_result = info.get("agent_result")

            if agent_result == "success":
                reward += 0.3

            elif agent_result == "failed":
                reward -= 0.4

        # ---------------- CRITICAL FIX REWARDS ---------------- #

        if prev_network == "down" and curr_network in ["slow", "normal"]:
            reward += 0.6 * weight

        if prev_service != "healthy" and curr_service == "healthy":
            reward += 0.8 * weight

        # ---------------- WRONG ACTION PENALTY ---------------- #

        if prev_cpu > 80 and action != "delegate_sre":
            reward -= 0.3

        if prev_network == "down" and action not in ["delegate_network", "restart_network"]:
            reward -= 0.3

        if prev_memory > 80 and action != "delegate_memory":
            reward -= 0.2

        # ---------------- ACTION COST (REALISM) ---------------- #

        action_cost = {
            "delegate_sre": -0.05,
            "delegate_memory": -0.04,
            "delegate_network": -0.05,
            "restart_service": -0.06,
            "rollback_deployment": -0.08,
        }

        reward += action_cost.get(action, 0)

        # ---------------- ANTI-SPAM ---------------- #

        if self.last_action == action:
            reward -= 0.2

        if action == "do_nothing" and not done:
            reward -= 0.4

        # ---------------- NO PROGRESS ---------------- #

        if not system_improved:
            reward -= 0.3

        # ---------------- TERMINAL BONUS ---------------- #

        if done:
            if system_healthy:
                if severity == "high":
                    reward += 6.0
                elif severity == "medium":
                    reward += 4.0
                else:
                    reward += 3.0
            else:
                reward -= 2.0  # failed episode penalty

        # ---------------- MEMORY ---------------- #

        self.last_action = action

        return reward