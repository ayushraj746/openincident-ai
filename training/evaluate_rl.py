from stable_baselines3 import PPO
from env.gym_wrapper import GymOpenIncidentEnv
import numpy as np

env = GymOpenIncidentEnv()
model = PPO.load("ppo_incident_model")

num_episodes = 20

all_rewards = []
all_steps = []

print("\n===== RAW EPISODE RESULTS =====\n")

for ep in range(num_episodes):
    obs, _ = env.reset()
    total_reward = 0
    steps = 0

    while True:
        action, _ = model.predict(obs)
        action = int(action)

        obs, reward, done, _, _ = env.step(action)

        total_reward += reward
        steps += 1

        if done:
            break

    all_rewards.append(total_reward)
    all_steps.append(steps)

    print(f"Episode {ep+1}: Reward={total_reward:.2f}, Steps={steps}")

# ---------------- FILTER LOGIC ---------------- #

valid_rewards = []
valid_steps = []

for r, s in zip(all_rewards, all_steps):
    if s <= 15:   # 🔥 remove unstable long episodes
        valid_rewards.append(r)
        valid_steps.append(s)

# ---------------- SUMMARY ---------------- #

print("\n===== OVERALL SUMMARY =====")
print(f"Avg Reward (All): {np.mean(all_rewards):.2f}")
print(f"Avg Steps  (All): {np.mean(all_steps):.2f}")

print("\n===== FILTERED SUMMARY (STABLE EPISODES) =====")

if len(valid_rewards) > 0:
    print(f"Avg Reward (Filtered): {np.mean(valid_rewards):.2f}")
    print(f"Avg Steps  (Filtered): {np.mean(valid_steps):.2f}")
    print(f"Episodes Used: {len(valid_rewards)}/{num_episodes}")
else:
    print("No valid episodes after filtering!")

# ---------------- BEST EPISODE ---------------- #

best_idx = np.argmax(all_rewards)

print("\n===== BEST EPISODE =====")
print(f"Episode: {best_idx + 1}")
print(f"Reward: {all_rewards[best_idx]:.2f}")
print(f"Steps : {all_steps[best_idx]}")