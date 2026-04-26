import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env.gym_wrapper import GymOpenIncidentEnv
from agents.commander import IncidentCommander


# ---------------- CORE EVAL FUNCTION ---------------- #

def run_episode(env, model=None, rule_agent=None):
    obs, _ = env.reset()
    raw_state = env.env.state.to_dict()

    total_reward = 0
    steps = 0

    trajectory = []

    while True:
        if model:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            action_name = env.action_map[action]

        elif rule_agent:
            action_name, _ = rule_agent.decide(raw_state)

        else:
            raise ValueError("No agent provided")

        # step
        raw_state, reward, done, info = env.env.step(action_name)
        obs = env._convert_state(raw_state)

        total_reward += reward
        steps += 1

        trajectory.append({
            "step": steps,
            "action": action_name,
            "reward": reward,
            "cpu": raw_state.get("cpu_usage"),
            "latency": raw_state.get("latency"),
        })

        if done:
            break

    return total_reward, steps, trajectory


# ---------------- MAIN EVALUATION ---------------- #

def evaluate_model(
    model_path="models/ppo_incident_model.zip",
    episodes=30,
    difficulties=["easy", "medium", "hard"],
):

    results = {}

    model = PPO.load(model_path)
    rule_agent = IncidentCommander()

    for diff in difficulties:
        print(f"\n===== Evaluating Difficulty: {diff.upper()} =====")

        env = GymOpenIncidentEnv(difficulty=diff, eval_mode=True)

        rl_rewards, rl_steps = [], []
        rule_rewards, rule_steps = [], []

        for _ in range(episodes):

            # RL run
            r, s, _ = run_episode(env, model=model)
            rl_rewards.append(r)
            rl_steps.append(s)

            # Rule run
            r2, s2, _ = run_episode(env, rule_agent=rule_agent)
            rule_rewards.append(r2)
            rule_steps.append(s2)

        # ---------------- METRICS ---------------- #

        results[diff] = {
            "rl_avg_reward": np.mean(rl_rewards),
            "rule_avg_reward": np.mean(rule_rewards),
            "rl_avg_steps": np.mean(rl_steps),
            "rule_avg_steps": np.mean(rule_steps),
            "rl_success_rate": np.mean([s < env.env.max_steps for s in rl_steps]),
            "rule_success_rate": np.mean([s < env.env.max_steps for s in rule_steps]),
        }

        print("\n--- RL ---")
        print(f"Reward: {results[diff]['rl_avg_reward']:.2f}")
        print(f"Steps : {results[diff]['rl_avg_steps']:.2f}")
        print(f"Success: {results[diff]['rl_success_rate']*100:.1f}%")

        print("\n--- RULE ---")
        print(f"Reward: {results[diff]['rule_avg_reward']:.2f}")
        print(f"Steps : {results[diff]['rule_avg_steps']:.2f}")
        print(f"Success: {results[diff]['rule_success_rate']*100:.1f}%")

    return results


# ---------------- VISUALIZATION ---------------- #

def plot_comparison(results):
    difficulties = list(results.keys())

    rl_rewards = [results[d]["rl_avg_reward"] for d in difficulties]
    rule_rewards = [results[d]["rule_avg_reward"] for d in difficulties]

    x = np.arange(len(difficulties))

    plt.figure()
    plt.bar(x - 0.2, rl_rewards, 0.4, label="RL")
    plt.bar(x + 0.2, rule_rewards, 0.4, label="Rule")

    plt.xticks(x, difficulties)
    plt.ylabel("Average Reward")
    plt.title("RL vs Rule-Based Comparison")
    plt.legend()

    plt.savefig("comparison.png")
    print("\n📊 Saved: comparison.png")


# ---------------- TRAJECTORY DEBUG ---------------- #

def debug_single_episode(model_path="models/ppo_incident_model.zip"):
    env = GymOpenIncidentEnv(difficulty="hard", eval_mode=True)
    model = PPO.load(model_path)

    reward, steps, trajectory = run_episode(env, model=model)

    print("\n===== TRAJECTORY =====\n")
    for t in trajectory:
        print(
            f"Step {t['step']}: Action={t['action']} | "
            f"Reward={t['reward']:.2f} | CPU={t['cpu']:.1f}"
        )

    print(f"\nTotal Reward: {reward:.2f}, Steps: {steps}")


# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    results = evaluate_model()
    plot_comparison(results)
    debug_single_episode()