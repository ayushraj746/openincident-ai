from env.environment import OpenIncidentEnv
from agents.commander import IncidentCommander
from agents.sre_agent import SREAgent
from agents.support_agent import SupportAgent
from agents.security_agent import SecurityAgent
from reward.grader import EpisodeGrader


class InferenceEngine:
    def __init__(self, difficulty: str = "medium"):
        self.env = OpenIncidentEnv(difficulty=difficulty)

        self.commander = IncidentCommander()
        self.sre = SREAgent()
        self.support = SupportAgent()
        self.security = SecurityAgent()

        # agent registry (scalable)
        self.agent_map = {
            "sre": self.sre,
            "support": self.support,
            "network": self.support,  # network handled by support
            "security": self.security,
        }

    def run_episode(self, verbose: bool = True):
        self.grader = EpisodeGrader()

        state = self.env.reset()

        if verbose:
            print("\n🚀 Starting New Episode")
            print("Initial State:", state)
            print("=" * 50)

        done = False

        while not done:

            # ---------------- COMMANDER DECISION ---------------- #

            agent_name, action, reason = self.commander.decide(state)

            # ---------------- AGENT ROUTING ---------------- #

            agent = self.agent_map.get(agent_name)

            if agent:
                execution = agent.execute(action, state)
            else:
                execution = {
                    "action": "do_nothing",
                    "status": "skipped",
                    "reason": f"Unknown agent: {agent_name}"
                }

            final_action = execution["action"]

            # ---------------- ENV STEP ---------------- #

            state, reward, done, _ = self.env.step(final_action)

            # ---------------- GRADER UPDATE ---------------- #

            self.grader.update(reward, final_action, done)

            # ---------------- LOGGING ---------------- #

            if verbose:
                print(f"🧠 Commander Decision:")
                print(f"   Agent: {agent_name}")
                print(f"   Action: {action}")
                print(f"   Reason: {reason}")

                print(f"⚙️ Execution:")
                print(f"   Final Action: {execution['action']}")
                print(f"   Status: {execution['status']}")
                print(f"   Exec Reason: {execution['reason']}")

                print(f"📊 State:")
                print(state)

                print(f"💰 Reward: {round(reward, 3)}")
                print("-" * 50)

        # ---------------- FINAL METRICS ---------------- #

        metrics = self.grader.get_metrics()

        if verbose:
            print("\n✅ Episode Finished!")
            print("📊 Metrics:")
            for k, v in metrics.items():
                print(f"   {k}: {v}")

        return metrics