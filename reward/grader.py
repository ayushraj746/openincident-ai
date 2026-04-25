class EpisodeGrader:
    def __init__(self):
        self.total_reward = 0
        self.steps = 0

        self.actions = []
        self.success = False

    def update(self, reward: float, action: str = None, done: bool = False):
        self.total_reward += reward
        self.steps += 1

        if action:
            self.actions.append(action)

        if done:
            self.success = True

    def get_metrics(self):

        unique_actions = len(set(self.actions)) if self.actions else 0

        # efficiency (fewer steps = better)
        efficiency = 1.0 / self.steps if self.steps > 0 else 0

        # stability (less repetition = better)
        repetition_penalty = len(self.actions) - unique_actions

        stability_score = max(0, 1 - (repetition_penalty / max(1, self.steps)))

        return {
            "total_reward": round(self.total_reward, 3),
            "steps": self.steps,
            "avg_reward": round(self.total_reward / max(1, self.steps), 3),

            # 🔥 NEW METRICS
            "success": self.success,
            "efficiency": round(efficiency, 3),
            "stability_score": round(stability_score, 3),
            "unique_actions": unique_actions,
        }