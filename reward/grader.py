class EpisodeGrader:
    def __init__(self):
        self.total_reward = 0
        self.steps = 0

    def update(self, reward: float):
        self.total_reward += reward
        self.steps += 1

    def get_metrics(self):
        return {
            "total_reward": self.total_reward,
            "steps": self.steps,
            "avg_reward": self.total_reward / max(1, self.steps),
        }