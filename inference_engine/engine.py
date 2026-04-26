import os

from env.environment import OpenIncidentEnv
from agents.commander import IncidentCommander
from agents.sre_agent import SREAgent
from agents.support_agent import SupportAgent
from agents.security_agent import SecurityAgent
from reward.grader import EpisodeGrader

from stable_baselines3 import PPO


class InferenceEngine:
    def __init__(self, difficulty="medium", mode="rule"):

        self.mode = mode
        self.model = None

        # ---------------- ENV ---------------- #
        self.env = OpenIncidentEnv(difficulty=difficulty)

        # ---------------- RL MODEL ---------------- #
        if mode == "rl":
            model_path = "ppo_incident_model"

            try:
                self.model = PPO.load(model_path)
                print("✅ RL model loaded")
            except:
                print("⚠️ RL model not found → fallback to rule-based")
                self.mode = "rule"

        # ---------------- AGENTS ---------------- #
        self.commander = IncidentCommander(rl_model=self.model)
        self.sre = SREAgent()
        self.support = SupportAgent()
        self.security = SecurityAgent()

        self.agent_map = {
            "sre": self.sre,
            "network": self.support,
            "support": self.support,
            "security": self.security,
        }

    # ---------------- AGENT ROUTING ---------------- #

    def _route_agent(self, action):
        if "sre" in action:
            return "sre"
        elif "network" in action:
            return "network"
        elif "memory" in action:
            return "sre"
        elif "security" in action:
            return "security"
        return "support"

    # ---------------- MAIN RUN ---------------- #

    def run_episode(self, verbose=True):

        grader = EpisodeGrader()
        state = self.env.reset()
        done = False

        step = 0

        while not done:
            step += 1

            # ---------------- RL OBS ---------------- #
            obs_vector = None
            if self.mode == "rl" and self.model is not None:
                try:
                    obs_vector = self.env.get_observation_vector()
                except:
                    obs_vector = None

            # ---------------- COMMANDER ---------------- #
            action, reason = self.commander.decide(state, obs_vector)

            # ---------------- AGENT ROUTING ---------------- #
            agent_key = self._route_agent(action)
            agent = self.agent_map.get(agent_key)

            if agent:
                execution = agent.execute(action, state)
                final_action = execution["action"]
            else:
                execution = {
                    "action": "do_nothing",
                    "status": "skipped",
                    "reason": "Unknown agent",
                    "impact": {}
                }
                final_action = "do_nothing"

            # ---------------- ENV STEP ---------------- #
            state, reward, done, _ = self.env.step(final_action)

            # 🔥 APPLY IMPACT
            if "impact" in execution:
                state.update(execution["impact"])

            # ---------------- GRADER ---------------- #
            grader.update(
                reward=reward,
                state=state,
                action=final_action,
                done=done
            )

            if verbose:
                print(f"Step {step} | Action={final_action} | Reward={round(reward,2)}")

        return grader.get_metrics()