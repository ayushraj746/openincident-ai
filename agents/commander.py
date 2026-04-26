from utils.hf_llm import query_llm


class IncidentCommander:
    def __init__(self, rl_model=None):
        self.history = []
        self.rl_model = rl_model

    # ---------------- STATE EXTRACTION ---------------- #

    def _extract_state(self, state: dict):
        if "metrics" in state:
            cpu = state["metrics"].get("processor_load")
            latency = state["metrics"].get("response_time")
            memory = state["system"].get("memory")
            network = state["system"].get("network")
            health = state["system"].get("service")

            context = state.get("context", {})
            severity = context.get("severity", "low")
        else:
            cpu = state.get("cpu_usage")
            latency = state.get("latency")
            memory = state.get("memory_usage")
            network = state.get("network_status")
            health = state.get("service_health")
            severity = state.get("severity", "low")

        return {
            "cpu": cpu or 50,
            "latency": latency or 500,
            "memory": memory or 50,
            "network": network or "normal",
            "health": health or "degraded",
            "severity": severity,
        }

    # ---------------- RULE SAFETY ---------------- #

    def _rule_safety(self, state, action):
        if state["network"] == "down" and action != "delegate_network":
            return "delegate_network", "Critical fix: network down"

        if state["cpu"] > 90 and action != "delegate_sre":
            return "delegate_sre", "Critical fix: CPU overload"

        return action, "Accepted"

    # ---------------- RL DECISION ---------------- #

    def _rl_decision(self, obs_vector):
        if self.rl_model is None:
            return None

        action, _ = self.rl_model.predict(obs_vector)
        return action

    # ---------------- LLM EXPLANATION ---------------- #

    def _llm_explain(self, state, action):

        # call LLM every 2 steps only (cost control)
        if len(self.history) % 2 != 0:
            return f"Rule-based fallback → action={action}"

        prompt = f"""
        You are an expert Site Reliability Engineer (SRE).

        System State:
        - CPU Usage: {state['cpu']}%
        - Latency: {state['latency']} ms
        - Memory: {state['memory']}%
        - Network: {state['network']}
        - Service Health: {state['health']}

        Task:
        Explain in 1–2 lines WHY action '{action}' is the best decision.
        Focus on root cause and system stability.
        """

        try:
            response = query_llm(prompt)

            # fallback if empty / error
            if not response or "error" in response.lower():
                return f"Fallback explanation → action={action}"

            return response.strip()

        except Exception as e:
            return f"LLM failed → action={action}"

    # ---------------- MAIN DECISION ---------------- #

    def decide(self, state: dict, obs_vector=None):

        extracted = self._extract_state(state)

        # ---------------- RL SUGGESTION ---------------- #
        rl_action = None
        if obs_vector is not None:
            rl_action = self._rl_decision(obs_vector)

        action_map = {
            0: "delegate_sre",
            1: "delegate_network",
            2: "delegate_memory",
            3: "rollback_deployment",
            4: "restart_service",
            5: "do_nothing",
        }

        if rl_action is not None:
            action = action_map.get(int(rl_action), "do_nothing")
        else:
            action = "do_nothing"

        # ---------------- RULE CORRECTION ---------------- #
        safe_action, _ = self._rule_safety(extracted, action)

        # ---------------- LLM EXPLANATION ---------------- #
        explanation = self._llm_explain(extracted, safe_action)

        # 🔥 DEBUG PRINT (VERY IMPORTANT)
        print(f"🧠 LLM Reason: {explanation}")

        # ---------------- HISTORY ---------------- #
        self.history.append({
            "state": state,
            "action": safe_action,
            "reason": explanation
        })

        return safe_action, explanation