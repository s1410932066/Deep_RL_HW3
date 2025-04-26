import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
np.random.seed(42)
n_arms = 10
n_rounds = 1000
n_sims = 1000
true_means = np.random.rand(n_arms)

# Epsilon-Greedy simulation
def run_epsilon_greedy(epsilon, true_means, n_sims, n_rounds):
    n_arms = len(true_means)
    cumulative = np.zeros((n_sims, n_rounds))
    for sim in range(n_sims):
        counts = np.zeros(n_arms)
        values = np.zeros(n_arms)
        total = 0
        for t in range(n_rounds):
            if np.random.rand() < epsilon:
                arm = np.random.randint(n_arms)
            else:
                arm = np.argmax(values)
            reward = 1 if np.random.rand() < true_means[arm] else 0
            counts[arm] += 1
            values[arm] += (reward - values[arm]) / counts[arm]
            total += reward
            cumulative[sim, t] = total / (t + 1)
    return cumulative.mean(axis=0)

avg_eps = run_epsilon_greedy(0.1, true_means, n_sims, n_rounds)
plt.figure()
plt.plot(avg_eps)
plt.xlabel('Rounds')
plt.ylabel('Average Cumulative Reward')
plt.title('Epsilon-Greedy Average Cumulative Reward')
plt.grid(True)
plt.show()
