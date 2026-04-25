class SecurityAgent:
    def __init__(self):
        self.supported_actions = {
            "investigate_security",
            "block_traffic",
            "reset_connections",
            "do_nothing"
        }

    def _safe_value(self, value, default):
        return value if value is not None else default

    def execute(self, action: str, state: dict):
        """
        Executes security-related actions

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
            latency = self._safe_value(state["metrics"].get("response_time"), 500)
            network = state["system"].get("network", "normal")
        else:
            latency = self._safe_value(state.get("latency"), 500)
            network = state.get("network_status", "normal")

        # ---------------- SECURITY DETECTION ---------------- #

        suspicious = False

        if latency > 1200:
            suspicious = True

        if network == "down":
            suspicious = True

        # ---------------- EXECUTION LOGIC ---------------- #

        if action == "investigate_security":
            if suspicious:
                return {
                    "action": action,
                    "status": "executed",
                    "reason": f"Suspicious activity detected (latency={latency}, network={network})"
                }
            else:
                return {
                    "action": "do_nothing",
                    "status": "skipped",
                    "reason": "No suspicious activity detected"
                }

        if action == "block_traffic":
            if suspicious:
                return {
                    "action": action,
                    "status": "executed",
                    "reason": "Blocking suspicious traffic"
                }
            else:
                return {
                    "action": "do_nothing",
                    "status": "skipped",
                    "reason": "No need to block traffic"
                }

        if action == "reset_connections":
            if network in ["down", "slow"]:
                return {
                    "action": action,
                    "status": "executed",
                    "reason": f"Resetting unstable connections ({network})"
                }
            else:
                return {
                    "action": "do_nothing",
                    "status": "skipped",
                    "reason": "Network stable"
                }

        return {
            "action": "do_nothing",
            "status": "executed",
            "reason": "No action required"
        }