from stable_baselines3 import PPO
from env.gym_wrapper import GymOpenIncidentEnv
import numpy as np


def evaluate_rl_model(
    model_path: str = "ppo_incident_model.zip",
    num_episodes: int = 20,
    difficulty: str = "medium",
    verbose: bool = True,
):
    """
    Evaluate trained RL model safely (non-breaking upgrade)

    Returns:
        dict:
            avg_reward
            avg_steps
            success_rate
    """

    # 🔥 Safe env init (eval mode ensures consistent scenario)
    env = GymOpenIncidentEnv(difficulty=difficulty, eval_mode=True)

    # 🔥 Safe model loading (handles .zip automatically)
    model = PPO.load(model_path)

    all_rewards = []
    all_steps = []
    successes = 0

    if verbose:
        print("\n===== RAW EPISODE RESULTS =====\n")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            obs, reward, done, _, _ = env.step(action)

            total_reward += reward
            steps += 1

            if done:
                break

        all_rewards.append(total_reward)
        all_steps.append(steps)

        # 🔥 Success condition (episode finished before max steps)
        if steps < env.env.max_steps:
            successes += 1

        if verbose:
            print(f"Episode {ep+1}: Reward={total_reward:.2f}, Steps={steps}")

    # ---------------- FINAL METRICS ---------------- #

    avg_reward = float(np.mean(all_rewards))
    avg_steps = float(np.mean(all_steps))
    success_rate = float(successes / num_episodes)

    if verbose:
        print("\n===== RL SUMMARY =====")
        print(f"Avg Reward   : {avg_reward:.2f}")
        print(f"Avg Steps    : {avg_steps:.2f}")
        print(f"Success Rate : {success_rate*100:.1f}%")

        # Best episode info (kept from your version ✔️)
        best_idx = int(np.argmax(all_rewards))
        print("\n===== BEST EPISODE =====")
        print(f"Episode: {best_idx + 1}")
        print(f"Reward : {all_rewards[best_idx]:.2f}")
        print(f"Steps  : {all_steps[best_idx]}")

    return {
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "success_rate": success_rate,
    }


# ---------------- CLI RUN (SAFE) ---------------- #

if __name__ == "__main__":
    evaluate_rl_model()