class SupportAgent:
    def __init__(self):
        self.supported_actions = {
            "restart_service",
            "restart_network",
            "delegate_network",
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
            network = self._safe(state["system"].get("network"), "normal")
            service = self._safe(state["system"].get("service"), "degraded")
            severity = state.get("context", {}).get("severity", "low")
        else:
            network = self._safe(state.get("network_status"), "normal")
            service = self._safe(state.get("service_health"), "degraded")
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

        # ---------------- NETWORK HANDLING ---------------- #

        if action in ["delegate_network", "restart_network"]:

            if network == "down":
                return self._result(
                    "restart_network",
                    "success",
                    "Network recovered (down → slow)"
                )

            elif network == "slow":
                return self._result(
                    "restart_network",
                    "success",
                    "Network stabilized (slow → normal)"
                )

            else:
                return self._result(
                    action,
                    "skipped",
                    "Network already stable"
                )

        # ---------------- SERVICE HANDLING ---------------- #

        if action == "restart_service":

            if service == "down":
                if network == "normal":
                    return self._result(
                        action,
                        "success",
                        "Service restarted → healthy"
                    )
                else:
                    return self._result(
                        action,
                        "success",
                        "Service restarted → degraded (network unstable)"
                    )

            elif service == "degraded":
                return self._result(
                    action,
                    "success",
                    "Service recovered → healthy"
                )

            else:
                return self._result(
                    action,
                    "skipped",
                    "Service already healthy"
                )

        return self._result(action, "executed", "No effect")

    # ---------------- RESULT ---------------- #

    def _result(self, action, status, reason):
        return {
            "action": action,
            "status": status,
            "reason": reason
        }