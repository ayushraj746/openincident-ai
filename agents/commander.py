class IncidentCommander:
    def __init__(self):
        self.history = []

    def _extract_state(self, state: dict):

        if "metrics" in state:
            cpu = state["metrics"].get("processor_load")
            latency = state["metrics"].get("response_time")
            memory = state["system"].get("memory")
            network = state["system"].get("network")
            health = state["system"].get("service")

            context = state.get("context", {})
            severity = context.get("severity", "low")
            agents = context.get("agents", [])
        else:
            cpu = state.get("cpu_usage")
            latency = state.get("latency")
            memory = state.get("memory_usage")
            network = state.get("network_status")
            health = state.get("service_health")
            severity = state.get("severity", "low")
            agents = state.get("available_agents", [])

        cpu = cpu if cpu is not None else 50
        latency = latency if latency is not None else 500
        memory = memory if memory is not None else 50
        network = network if network is not None else "normal"
        health = health if health is not None else "degraded"

        return cpu, latency, memory, network, health, severity, agents

    def _repeat_penalty(self, action):
        recent = [h["action"] for h in self.history[-3:]]
        return recent.count(action)

    def _no_progress(self, state):
        if len(self.history) < 2:
            return False

        prev_state = self.history[-1]["state"]

        return (
            prev_state.get("cpu_usage") == state.get("cpu_usage") and
            prev_state.get("latency") == state.get("latency") and
            prev_state.get("memory_usage") == state.get("memory_usage") and
            prev_state.get("network_status") == state.get("network_status") and
            prev_state.get("service_health") == state.get("service_health")
        )

    def decide(self, state: dict):

        cpu, latency, memory, network, health, severity, agents = self._extract_state(state)

        decisions = []

        # ---------------- 🔥 TERMINATION FIRST (VERY IMPORTANT) ---------------- #

        if (
            cpu < 65
            and latency < 750
            and memory < 65
            and network == "normal"
            and health == "healthy"
        ):
            return "sre", "do_nothing", "System stable — stopping actions"

        # ---------------- ROOT CAUSE FIRST ---------------- #

        if network == "down":
            decisions.append(("network", "delegate_network", 1.3, "Fixing network first"))

        if cpu > 80:
            decisions.append(("sre", "delegate_sre", 1.1, f"High CPU: {cpu}%"))

        if memory > 80:
            decisions.append(("sre", "delegate_memory", 1.0, f"High memory: {memory}%"))

        if latency > 800:
            decisions.append(("sre", "delegate_sre", 0.95, f"High latency: {latency}ms"))

        # ---------------- FINAL RECOVERY ---------------- #

        if network == "slow":
            decisions.append(("network", "delegate_network", 1.2, "Stabilizing network"))

        if health == "degraded" and network == "normal":
            decisions.append(("support", "restart_service", 1.1, "Final service recovery"))

        # ---------------- SERVICE LAST ---------------- #

        if health == "down":
            base_score = 0.5

            if network == "down":
                base_score = 0.3

            repeat_count = self._repeat_penalty("restart_service")
            if repeat_count > 0:
                base_score *= (1 / (1 + repeat_count))

            decisions.append(("support", "restart_service", base_score, "Service recovery"))

        # ---------------- NO PROGRESS ---------------- #

        if self._no_progress(state):
            decisions.append(("sre", "delegate_sre", 1.4, "No progress → forcing infra fix"))

        # ---------------- SEVERITY ---------------- #

        severity_weight = {
            "low": 1.0,
            "medium": 1.2,
            "high": 1.5,
        }

        weight = severity_weight.get(severity, 1.0)

        updated = []
        for agent, action, score, reason in decisions:
            repeat = self._repeat_penalty(action)
            score = score / (1 + repeat)
            updated.append((agent, action, score * weight, reason))

        decisions = updated

        # ---------------- FILTER ---------------- #

        decisions = [d for d in decisions if d[0] in agents or not agents]

        # ---------------- FALLBACK ---------------- #

        if not decisions:
            # 🔥 smarter fallback (avoid useless loops)
            if latency > 700:
                return "sre", "delegate_sre", "Fallback: latency optimization"
            return "sre", "do_nothing", "No meaningful action available"

        # ---------------- SELECT BEST ---------------- #

        best = max(decisions, key=lambda x: x[2])
        agent, action, score, reason = best

        self.history.append({
            "state": state,
            "action": action,
            "reason": reason
        })

        return agent, action, f"{reason} | confidence={round(score, 2)}"