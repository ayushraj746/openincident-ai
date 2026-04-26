import os

from env.environment import OpenIncidentEnv
from env.gym_wrapper import GymOpenIncidentEnv
from agents.commander import IncidentCommander
from agents.sre_agent import SREAgent
from agents.support_agent import SupportAgent
from agents.security_agent import SecurityAgent
from reward.grader import EpisodeGrader

from stable_baselines3 import PPO


class InferenceEngine:
    def __init__(self, difficulty: str = "medium", use_rl: bool = False):

        self.use_rl = use_rl
        self.model = None

        # ---------------- SAFE RL LOADING ---------------- #
        if use_rl:
            model_path = "models/ppo_incident_model.zip"

            if os.path.exists(model_path):
                print("✅ RL model loaded")
                self.env = GymOpenIncidentEnv(difficulty=difficulty, eval_mode=True)
                self.model = PPO.load(model_path)
            else:
                print("⚠️ RL model NOT found → fallback to rule-based")
                self.use_rl = False
                self.env = OpenIncidentEnv(difficulty=difficulty)
        else:
            self.env = OpenIncidentEnv(difficulty=difficulty)

        # ---------------- AGENTS ---------------- #
        self.commander = IncidentCommander()
        self.sre = SREAgent()
        self.support = SupportAgent()
        self.security = SecurityAgent()

    # ---------------- AGENT ROUTING ---------------- #

    def _route_agent(self, action):
        if "sre" in action:
            return self.sre
        elif "network" in action or "service" in action:
            return self.support
        elif "security" in action or "block" in action:
            return self.security
        return None

    # ---------------- MAIN RUN ---------------- #

    def run_episode(self, verbose=True, return_trajectory=False):

        grader = EpisodeGrader()
        trajectory = []

        if self.use_rl:
            obs, _ = self.env.reset()
        else:
            state = self.env.reset()

        done = False
        step = 0

        while not done:
            step += 1

            # ---------------- RL MODE ---------------- #
            if self.use_rl and self.model is not None:

                action_idx, _ = self.model.predict(obs, deterministic=True)
                action = self.env.action_map[int(action_idx)]

                obs, reward, done, _, _ = self.env.step(action)

                state_dict = {}

            # ---------------- RULE MODE ---------------- #
            else:
                action, reason = self.commander.decide(state)

                agent = self._route_agent(action)

                if agent:
                    execution = agent.execute(action, state)
                    final_action = execution["action"]
                else:
                    final_action = "do_nothing"

                state, reward, done, _ = self.env.step(final_action)

                state_dict = state

            # ---------------- GRADER ---------------- #
            grader.update(reward, state_dict, action, done)

            trajectory.append({
                "step": step,
                "action": action,
                "reward": reward
            })

            if verbose:
                print(f"Step {step} | Action={action} | Reward={round(reward,2)}")

        metrics = grader.get_metrics()

        if return_trajectory:
            return metrics, trajectory

        return metrics