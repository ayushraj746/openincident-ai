class IncidentCommander:
    def __init__(self):
        pass

    def decide(self, state: dict):
        """
        Returns: (agent, action)
        """

        cpu = state.get("cpu_usage")
        latency = state.get("latency")
        memory = state.get("memory_usage")
        network = state.get("network_status")
        health = state.get("service_health")

        # ---------------- HANDLE MISSING VALUES ---------------- #

        if cpu is None:
            cpu = 50

        if latency is None:
            latency = 500

        if memory is None:
            memory = 50

        if network is None:
            network = "normal"

        # ---------------- DECISION LOGIC ---------------- #

        # 🔥 Priority 1: Critical service down
        if health == "down":
            return "support", "restart_service"

        # 🔥 Priority 2: Network issues
        if network == "down":
            return "support", "restart_network"

        # 🔥 Priority 3: High CPU
        if cpu > 80:
            return "sre", "scale_resources"

        # 🔥 Priority 4: High Memory
        if memory > 80:
            return "sre", "clear_cache"

        # 🔥 Default
        return "sre", "do_nothing"