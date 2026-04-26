from stable_baselines3 import PPO
from env.gym_wrapper import GymOpenIncidentEnv

# create env
env = GymOpenIncidentEnv()

# create model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
)

# train
model.learn(total_timesteps=50000)

# save model
model.save("ppo_incident_model")

print("✅ Training Complete")