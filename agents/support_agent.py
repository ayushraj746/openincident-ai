class SupportAgent:
    def __init__(self):
        self.supported_actions = {
            "restart_service",
            "restart_network",
            "delegate_network",   # 🔥 NEW (VERY IMPORTANT)
            "do_nothing"
        }

    def _safe_value(self, value, default):
        return value if value is not None else default

    def execute(self, action: str, state: dict):
        """
        Executes support-level actions safely

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
            network = state["system"].get("network", "normal")
            service = state["system"].get("service", "degraded")
        else:
            network = state.get("network_status", "normal")
            service = state.get("service_health", "degraded")

        network = self._safe_value(network, "normal")
        service = self._safe_value(service, "degraded")

        # ---------------- INTELLIGENT EXECUTION ---------------- #

        # 🔥 NETWORK HANDLING (CRITICAL FIX)
        if action == "delegate_network":
            if network in ["down", "slow"]:
                return {
                    "action": "restart_network",
                    "status": "executed",
                    "reason": f"Handling network issue ({network}) → restarting network"
                }
            else:
                return {
                    "action": "do_nothing",
                    "status": "skipped",
                    "reason": "Network is stable, no action needed"
                }

        # Service restart logic
        if action == "restart_service":
            if service == "down":
                return {
                    "action": action,
                    "status": "executed",
                    "reason": "Service is down, restarting service"
                }
            else:
                return {
                    "action": "do_nothing",
                    "status": "skipped",
                    "reason": f"Service not down (current state: {service})"
                }

        # Direct network restart
        if action == "restart_network":
            if network in ["down", "slow"]:
                return {
                    "action": action,
                    "status": "executed",
                    "reason": f"Network issue detected ({network}), restarting network"
                }
            else:
                return {
                    "action": "do_nothing",
                    "status": "skipped",
                    "reason": "Network already stable"
                }

        # Default fallback
        return {
            "action": "do_nothing",
            "status": "executed",
            "reason": "No action required"
        }