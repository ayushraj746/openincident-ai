from env.environment import OpenIncidentEnv
from agents.commander import IncidentCommander
from agents.sre_agent import SREAgent
from agents.support_agent import SupportAgent

from training.evaluate_rl import evaluate_rl_model

import numpy as np

NUM_EPISODES = 20


# -------- RULE BASED -------- #
def evaluate_rule_based(num_episodes=20):
    rewards = []
    steps_list = []
    successes = 0

    for _ in range(num_episodes):
        env = OpenIncidentEnv(eval_mode=True)

        commander = IncidentCommander()
        sre = SREAgent()
        support = SupportAgent()

        state = env.reset()
        total_reward = 0
        steps = 0

        done = False

        while not done:
            agent_name, action, _ = commander.decide(state)

            if agent_name == "sre":
                action = sre.execute(action, state)["action"]
            elif agent_name == "support":
                action = support.execute(action, state)["action"]

            state, reward, done, _ = env.step(action)

            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        steps_list.append(steps)

        # success condition
        if steps < env.max_steps:
            successes += 1

    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_steps": float(np.mean(steps_list)),
        "success_rate": float(successes / num_episodes),
    }


# -------- RUN COMPARISON -------- #

rule_results = evaluate_rule_based(NUM_EPISODES)

rl_results = evaluate_rl_model(
    model_path="ppo_incident_model.zip",
    num_episodes=NUM_EPISODES,
    verbose=False,
)


# -------- PRINT RESULTS -------- #

print("\n===== FINAL COMPARISON =====\n")

print("---- Rule-Based System ----")
print(f"Avg Reward   : {rule_results['avg_reward']:.2f}")
print(f"Avg Steps    : {rule_results['avg_steps']:.2f}")
print(f"Success Rate : {rule_results['success_rate']*100:.1f}%")

print("\n---- RL Agent ----")
print(f"Avg Reward   : {rl_results['avg_reward']:.2f}")
print(f"Avg Steps    : {rl_results['avg_steps']:.2f}")
print(f"Success Rate : {rl_results['success_rate']*100:.1f}%")

# -------- IMPROVEMENT SUMMARY -------- #

print("\n===== IMPROVEMENT SUMMARY =====\n")

reward_diff = rl_results["avg_reward"] - rule_results["avg_reward"]
steps_diff = rule_results["avg_steps"] - rl_results["avg_steps"]
success_diff = rl_results["success_rate"] - rule_results["success_rate"]

print(f"Reward Change   : {reward_diff:+.2f}")
print(f"Steps Change    : {steps_diff:+.2f} (positive = RL faster)")
print(f"Success Change  : {success_diff*100:+.1f}%")