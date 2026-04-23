class SupportAgent:
    def execute(self, action: str) -> str:
        """
        Handles user-facing issues
        """

        allowed_actions = [
            "restart_service",
            "restart_network"
        ]

        if action in allowed_actions:
            return action

        return "do_nothing"