from env.gym_wrapper import GymOpenIncidentEnv

env = GymOpenIncidentEnv()

obs, _ = env.reset()
print("Initial Obs:", obs)

for _ in range(5):
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)

    print("Action:", action)
    print("Obs:", obs)
    print("Reward:", reward)
    print("-" * 30)

    if done:
        print("✅ Finished early")
        break