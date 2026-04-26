class SREAgent:
    def __init__(self):
        self.supported_actions = {
            "delegate_sre",
            "delegate_memory",
            "rollback_deployment",
            "do_nothing"
        }

    def _safe_value(self, value, default):
        return value if value is not None else default

    def execute(self, action: str, state: dict):
        """
        Executes infrastructure-related actions safely

        Returns:
            {
                "action": str,
                "status": "executed" | "skipped",
                "reason": str
            }
        """

        # ---------------- VALIDATION ---------------- #

        if action not in self.supported_actions:
            return {
                "action": "do_nothing",
                "status": "skipped",
                "reason": f"Unsupported action: {action}"
            }

        # ---------------- STATE EXTRACTION ---------------- #

        if "metrics" in state:
            cpu = self._safe_value(state["metrics"].get("processor_load"), 50)
            latency = self._safe_value(state["metrics"].get("response_time"), 500)
            memory = self._safe_value(state["system"].get("memory"), 50)
        else:
            cpu = self._safe_value(state.get("cpu_usage"), 50)
            latency = self._safe_value(state.get("latency"), 500)
            memory = self._safe_value(state.get("memory_usage"), 50)

        # ---------------- INTELLIGENT EXECUTION ---------------- #

        # 🔥 INFRA SCALING (CPU OR LATENCY)
        if action == "delegate_sre":
            if cpu > 75 or latency > 800:
                return {
                    "action": action,
                    "status": "executed",
                    "reason": f"Scaling infra (cpu={round(cpu,2)}%, latency={round(latency,2)}ms)"
                }
            else:
                return {
                    "action": "do_nothing",
                    "status": "skipped",
                    "reason": f"Infra stable (cpu={round(cpu,2)}%, latency={round(latency,2)}ms)"
                }

        # 🔥 MEMORY CLEANUP
        if action == "delegate_memory":
            if memory > 70:
                return {
                    "action": action,
                    "status": "executed",
                    "reason": f"Clearing cache (memory={round(memory,2)}%)"
                }
            else:
                return {
                    "action": "do_nothing",
                    "status": "skipped",
                    "reason": f"Memory normal ({round(memory,2)}%)"
                }

        # 🔥 ROLLBACK (ALWAYS VALID)
        if action == "rollback_deployment":
            return {
                "action": action,
                "status": "executed",
                "reason": "Rolling back deployment to stable version"
            }

        # Default fallback
        return {
            "action": "do_nothing",
            "status": "executed",
            "reason": "No infra action required"
        }