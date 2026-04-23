from env.environment import OpenIncidentEnv
from agents.commander import IncidentCommander
from agents.sre_agent import SREAgent
from agents.support_agent import SupportAgent
from reward.grader import EpisodeGrader


class InferenceEngine:
    def __init__(self, difficulty: str = "medium"):
        self.env = OpenIncidentEnv(difficulty=difficulty)

        self.commander = IncidentCommander()
        self.sre = SREAgent()
        self.support = SupportAgent()

    def run_episode(self, verbose: bool = True):
        # 🔥 IMPORTANT: reset grader every episode
        self.grader = EpisodeGrader()

        state = self.env.reset()

        if verbose:
            print("Initial State:", state)

        done = False

        while not done:
            # Commander decides
            agent_name, action = self.commander.decide(state)

            # Delegation
            if agent_name == "sre":
                executed_action = self.sre.execute(action)
            elif agent_name == "support":
                executed_action = self.support.execute(action)
            else:
                executed_action = "do_nothing"

            # Environment step
            state, reward, done, _ = self.env.step(executed_action)

            # Update metrics
            self.grader.update(reward)

            if verbose:
                print(f"Commander → {agent_name} : {action}")
                print("Executed Action:", executed_action)
                print("State:", state)
                print("Reward:", reward)
                print("-" * 40)

        metrics = self.grader.get_metrics()

        if verbose:
            print("✅ Episode Finished!")
            print("📊 Metrics:", metrics)

        return metrics