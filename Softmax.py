import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
np.random.seed(42)
n_arms = 10
n_rounds = 1000
n_sims = 1000
true_means = np.random.rand(n_arms)

# Softmax simulation
def run_softmax(tau, true_means, n_sims, n_rounds):
    cumulative = np.zeros((n_sims, n_rounds))
    for sim in range(n_sims):
        counts = np.zeros(n_arms)
        values = np.zeros(n_arms)
        total = 0
        for t in range(n_rounds):
            exp_vals = np.exp(values / tau)
            probs = exp_vals / exp_vals.sum()
            arm = np.random.choice(n_arms, p=probs)
            r = 1 if np.random.rand() < true_means[arm] else 0
            counts[arm] += 1
            values[arm] += (r - values[arm]) / counts[arm]
            total += r
            cumulative[sim, t] = total / (t+1)
    return cumulative.mean(axis=0)

avg_soft = run_softmax(0.1, true_means, n_sims, n_rounds)
plt.figure()
plt.plot(avg_soft)
plt.xlabel('Rounds')
plt.ylabel('Average Cumulative Reward')
plt.title('Softmax Average Cumulative Reward')
plt.grid(True)
plt.show()
