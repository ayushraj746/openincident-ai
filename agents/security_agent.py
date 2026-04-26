class SecurityAgent:
    def __init__(self):
        self.supported_actions = {
            "investigate_security",
            "block_traffic",
            "reset_connections",
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
            latency = self._safe(state["metrics"].get("response_time"), 500)
            network = self._safe(state["system"].get("network"), "normal")
        else:
            latency = self._safe(state.get("latency"), 500)
            network = self._safe(state.get("network_status"), "normal")

        # ---------------- THREAT DETECTION ---------------- #

        threat = False

        if latency > 1200:
            threat = True

        if network == "down":
            threat = True

        # ---------------- DO NOTHING ---------------- #

        if action == "do_nothing":
            return self._result(action, "executed", "No security action")

        # ---------------- INVESTIGATION ---------------- #

        if action == "investigate_security":

            if threat:
                return self._result(
                    action,
                    "success",
                    "Threat detected based on latency/network"
                )
            else:
                return self._result(
                    action,
                    "skipped",
                    "No threat signals"
                )

        # ---------------- BLOCK TRAFFIC ---------------- #

        if action == "block_traffic":

            if threat:
                return self._result(
                    action,
                    "success",
                    "Traffic blocked to mitigate threat"
                )
            else:
                return self._result(
                    action,
                    "skipped",
                    "No threat to block"
                )

        # ---------------- RESET CONNECTIONS ---------------- #

        if action == "reset_connections":

            if network in ["down", "slow"]:
                return self._result(
                    action,
                    "success",
                    "Connections reset → network stabilizing"
                )
            else:
                return self._result(
                    action,
                    "skipped",
                    "Network already stable"
                )

        return self._result(action, "executed", "No effect")

    # ---------------- RESULT ---------------- #

    def _result(self, action, status, reason):
        return {
            "action": action,
            "status": status,
            "reason": reason
        }