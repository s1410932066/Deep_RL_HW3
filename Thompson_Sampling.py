import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
np.random.seed(42)
n_arms = 10
n_rounds = 1000
n_sims = 1000
true_means = np.random.rand(n_arms)

# Thompson Sampling simulation
def run_thompson(true_means, n_sims, n_rounds):
    cumulative = np.zeros((n_sims, n_rounds))
    for sim in range(n_sims):
        succ = np.zeros(n_arms)
        fail = np.zeros(n_arms)
        total = 0
        for t in range(n_rounds):
            theta = np.random.beta(succ+1, fail+1)
            arm = np.argmax(theta)
            r = 1 if np.random.rand() < true_means[arm] else 0
            succ[arm] += r
            fail[arm] += 1 - r
            total += r
            cumulative[sim, t] = total / (t+1)
    return cumulative.mean(axis=0)

avg_ts = run_thompson(true_means, n_sims, n_rounds)
plt.figure()
plt.plot(avg_ts)
plt.xlabel('Rounds')
plt.ylabel('Average Cumulative Reward')
plt.title('Thompson Sampling Average Cumulative Reward')
plt.grid(True)
plt.show()
