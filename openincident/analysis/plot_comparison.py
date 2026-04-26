import matplotlib.pyplot as plt

labels = ["Rule-Based", "RL"]

rewards = [5.19, 6.62]
steps = [7.53, 9.50]

# Reward graph
plt.figure()
plt.bar(labels, rewards)
plt.title("Reward Comparison (Higher is Better)")
plt.ylabel("Reward")
plt.show()

# Steps graph
plt.figure()
plt.bar(labels, steps)
plt.title("Steps Comparison (Lower is Faster)")
plt.ylabel("Steps")
plt.show()