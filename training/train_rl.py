import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from env.gym_wrapper import GymOpenIncidentEnv


# ---------------- CUSTOM CALLBACK ---------------- #

class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_rewards = 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]

        self.current_rewards += reward

        if done:
            self.episode_rewards.append(self.current_rewards)
            self.current_rewards = 0

        return True


# ---------------- ENV ---------------- #

env = GymOpenIncidentEnv(difficulty="medium")

# ---------------- MODEL ---------------- #

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1,
)

# ---------------- TRAIN ---------------- #

callback = RewardLoggerCallback()

model.learn(
    total_timesteps=50000,
    callback=callback,
)

# ---------------- SAVE MODEL ---------------- #

os.makedirs("models", exist_ok=True)
model.save("models/ppo_incident_model")

# ---------------- PLOT REWARD ---------------- #

rewards = callback.episode_rewards

plt.figure()
plt.plot(rewards)
plt.title("Training Reward Curve")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("training_reward.png")

print("✅ Training Complete")
print("📈 Reward graph saved as training_reward.png")