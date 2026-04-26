class EpisodeGrader:
    def __init__(self):
        self.total_reward = 0
        self.steps = 0

        self.actions = []
        self.success = False

        # tracking
        self.state_history = []
        self.agent_usage = {}
        self.recovery_step = None

    # ---------------- SAFE EXTRACT ---------------- #

    def _extract(self, state):
        if not state:
            return {
                "cpu": 100,
                "latency": 1000,
                "memory": 100,
                "network": "down",
                "service": "down",
                "severity": "low"
            }

        if "metrics" in state:
            cpu = state["metrics"].get("processor_load", 100)
            latency = state["metrics"].get("response_time", 1000)
            memory = state["system"].get("memory", 100)
            network = state["system"].get("network", "down")
            service = state["system"].get("service", "down")
            severity = state.get("context", {}).get("severity", "low")
        else:
            cpu = state.get("cpu_usage", 100)
            latency = state.get("latency", 1000)
            memory = state.get("memory_usage", 100)
            network = state.get("network_status", "down")
            service = state.get("service_health", "down")
            severity = state.get("severity", "low")

        return {
            "cpu": cpu if cpu is not None else 100,
            "latency": latency if latency is not None else 1000,
            "memory": memory if memory is not None else 100,
            "network": network or "down",
            "service": service or "down",
            "severity": severity or "low"
        }

    # ---------------- UPDATE ---------------- #

    def update(self, reward: float, state: dict = None, action: str = None, done: bool = False):
        self.total_reward += reward
        self.steps += 1

        if action:
            self.actions.append(action)

            agent = action.split("_")[-1] if "_" in action else action
            self.agent_usage[agent] = self.agent_usage.get(agent, 0) + 1

        if state:
            extracted = self._extract(state)
            self.state_history.append(extracted)

            if self.recovery_step is None:
                if self._is_healthy(extracted):
                    self.recovery_step = self.steps

        if done and self.recovery_step is not None:
            self.success = True

    # ---------------- HEALTH CHECK ---------------- #

    def _is_healthy(self, s):
        return (
            s["cpu"] < 60
            and s["latency"] < 600
            and s["memory"] < 60
            and s["network"] == "normal"
            and s["service"] == "healthy"
        )

    # ---------------- METRICS ---------------- #

    def get_metrics(self):
        unique_actions = len(set(self.actions)) if self.actions else 0

        efficiency = 1.0 / self.steps if self.steps > 0 else 0

        repetition_penalty = len(self.actions) - unique_actions
        stability_score = max(0, 1 - (repetition_penalty / max(1, self.steps)))

        recovery_speed = (
            self.recovery_step / self.steps if self.recovery_step else 0
        )

        agent_diversity = len(self.agent_usage)

        robustness = 1 if self.success else 0

        # severity-aware scoring
        severity_weight = 1
        if self.state_history:
            severity = self.state_history[0]["severity"]
            if severity == "high":
                severity_weight = 2
            elif severity == "medium":
                severity_weight = 1.5

        weighted_reward = self.total_reward * severity_weight

        return {
            "total_reward": round(self.total_reward, 3),
            "weighted_reward": round(weighted_reward, 3),
            "steps": self.steps,
            "avg_reward": round(self.total_reward / max(1, self.steps), 3),

            "success": self.success,
            "efficiency": round(efficiency, 3),
            "stability_score": round(stability_score, 3),
            "unique_actions": unique_actions,

            "recovery_speed": round(recovery_speed, 3),
            "agent_diversity": agent_diversity,
            "robustness": robustness,
            "agent_usage": self.agent_usage,
        }