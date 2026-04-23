from env.environment import OpenIncidentEnv
from agents.commander import IncidentCommander
from agents.sre_agent import SREAgent
from agents.support_agent import SupportAgent

from stable_baselines3 import PPO
from env.gym_wrapper import GymOpenIncidentEnv
import numpy as np

NUM_EPISODES = 20


# -------- RULE BASED -------- #
def run_rule_based():
    rewards = []
    steps_list = []

    for _ in range(NUM_EPISODES):
        # 🔥 FIX: eval_mode=True for fair comparison
        env = OpenIncidentEnv(eval_mode=True)

        commander = IncidentCommander()
        sre = SREAgent()
        support = SupportAgent()

        state = env.reset()
        total_reward = 0
        steps = 0

        done = False

        while not done:
            agent_name, action = commander.decide(state)

            if agent_name == "sre":
                action = sre.execute(action)
            elif agent_name == "support":
                action = support.execute(action)

            state, reward, done, _ = env.step(action)

            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        steps_list.append(steps)

    return rewards, steps_list


# -------- RL -------- #
def run_rl():
    # 🔥 FIX: eval_mode=True
    env = GymOpenIncidentEnv(eval_mode=True)
    model = PPO.load("ppo_incident_model")

    rewards = []
    steps_list = []

    for _ in range(NUM_EPISODES):
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

        rewards.append(total_reward)
        steps_list.append(steps)

    return rewards, steps_list


# -------- FILTER -------- #
def filter_data(rewards, steps):
    filtered_r = []
    filtered_s = []

    for r, s in zip(rewards, steps):
        if 3 <= s <= 15:   # 🔥 better filtering (ignore too short + too long)
            filtered_r.append(r)
            filtered_s.append(s)

    return filtered_r, filtered_s


# -------- RUN -------- #

rule_rewards, rule_steps = run_rule_based()
rl_rewards, rl_steps = run_rl()

rule_rewards_f, rule_steps_f = filter_data(rule_rewards, rule_steps)
rl_rewards_f, rl_steps_f = filter_data(rl_rewards, rl_steps)


print("\n===== FINAL COMPARISON (FILTERED) =====")

print(f"Rule-Based → Reward: {np.mean(rule_rewards_f):.2f}, Steps: {np.mean(rule_steps_f):.2f}")
print(f"RL Agent   → Reward: {np.mean(rl_rewards_f):.2f}, Steps: {np.mean(rl_steps_f):.2f}")


# -------- EXTRA DEBUG (VERY HELPFUL) -------- #

print("\n===== RAW STATS =====")
print(f"Rule Avg Steps (All): {np.mean(rule_steps):.2f}")
print(f"RL Avg Steps (All):   {np.mean(rl_steps):.2f}")

print(f"Valid Episodes Used: Rule={len(rule_steps_f)}, RL={len(rl_steps_f)}")