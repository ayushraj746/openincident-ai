from stable_baselines3 import PPO
from env.gym_wrapper import GymOpenIncidentEnv

# create environment
env = GymOpenIncidentEnv()

# load trained model
model = PPO.load("ppo_incident_model")

# reset env
obs, _ = env.reset()

for step in range(20):
    # predict action
    action, _ = model.predict(obs)

    # 🔥 IMPORTANT FIX (numpy → int)
    action = int(action)

    # take step
    obs, reward, done, _, _ = env.step(action)

    print(f"Step: {step + 1}")
    print("Action:", action)
    print("Reward:", reward)
    print("-" * 30)

    if done:
        print("✅ Finished")
        break