import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
np.random.seed(42)
n_arms = 10
n_rounds = 1000
n_sims = 1000
true_means = np.random.rand(n_arms)

# UCB simulation
def run_ucb(c, true_means, n_sims, n_rounds):
    cumulative = np.zeros((n_sims, n_rounds))
    for sim in range(n_sims):
        counts = np.zeros(n_arms)
        values = np.zeros(n_arms)
        total = 0
        # Pull each arm once
        for a in range(n_arms):
            r = 1 if np.random.rand() < true_means[a] else 0
            counts[a] = 1
            values[a] = r
            total += r
            cumulative[sim, a] = total / (a+1)
        for t in range(n_arms, n_rounds):
            ucb_vals = values + c * np.sqrt(np.log(t+1) / counts)
            arm = np.argmax(ucb_vals)
            r = 1 if np.random.rand() < true_means[arm] else 0
            counts[arm] += 1
            values[arm] += (r - values[arm]) / counts[arm]
            total += r
            cumulative[sim, t] = total / (t+1)
    return cumulative.mean(axis=0)

avg_ucb = run_ucb(2, true_means, n_sims, n_rounds)
plt.figure()
plt.plot(avg_ucb)
plt.xlabel('Rounds')
plt.ylabel('Average Cumulative Reward')
plt.title('UCB Average Cumulative Reward')
plt.grid(True)
plt.show()
