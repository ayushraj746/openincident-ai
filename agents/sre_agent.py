class SREAgent:
    def execute(self, action: str) -> str:
        """
        Handles infrastructure-related actions
        """

        allowed_actions = [
            "scale_resources",
            "clear_cache",
            "do_nothing"
        ]

        if action in allowed_actions:
            return action

        return "do_nothing"