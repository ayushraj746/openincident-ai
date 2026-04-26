import random


class SREAgent:
    def __init__(self):
        self.supported_actions = {
            "delegate_sre",
            "delegate_memory",
            "rollback_deployment",
            "do_nothing"
        }

    def _safe(self, value, default):
        return value if value is not None else default

    # ---------------- EXECUTION ---------------- #

    def execute(self, action: str, state: dict):

        if action not in self.supported_actions:
            return self._result("do_nothing", "skipped", "Unsupported action")

        # ---------------- EXTRACT ---------------- #

        if "metrics" in state:
            cpu = self._safe(state["metrics"].get("processor_load"), 50)
            latency = self._safe(state["metrics"].get("response_time"), 500)
            memory = self._safe(state["system"].get("memory"), 50)

            severity = state.get("context", {}).get("severity", "low")
        else:
            cpu = self._safe(state.get("cpu_usage"), 50)
            latency = self._safe(state.get("latency"), 500)
            memory = self._safe(state.get("memory_usage"), 50)

            severity = state.get("severity", "low")

        # ---------------- DO NOTHING ---------------- #

        if action == "do_nothing":
            return self._result(action, "executed", "No action needed")

        # ---------------- SEVERITY MULTIPLIER ---------------- #

        severity_factor = 1.0
        if severity == "high":
            severity_factor = 1.5
        elif severity == "medium":
            severity_factor = 1.2

        # ---------------- LOGIC ---------------- #

        # 🔥 INFRA SCALING
        if action == "delegate_sre":

            if cpu > 75 or latency > 800:

                cpu_reduction = 20 * severity_factor
                latency_reduction = 300 * severity_factor

                return self._result(
                    action,
                    "success",
                    f"Scaled infra (cpu↓{round(cpu_reduction)}, latency↓{round(latency_reduction)})"
                )

            return self._result(action, "skipped", "Infra already stable")

        # 🔥 MEMORY CLEANUP
        if action == "delegate_memory":

            if memory > 70:

                mem_reduction = 20 * severity_factor

                return self._result(
                    action,
                    "success",
                    f"Memory reduced (↓{round(mem_reduction)})"
                )

            return self._result(action, "skipped", "Memory normal")

        # 🔥 ROLLBACK
        if action == "rollback_deployment":

            return self._result(
                action,
                "success",
                "Rollback applied to stabilize system"
            )

        return self._result(action, "executed", "No effect")

    # ---------------- RESULT ---------------- #

    def _result(self, action, status, reason):
        return {
            "action": action,
            "status": status,
            "reason": reason
        }