## 1. Epsilon-Greedy

**(1) Algorithm Formula**

$$
\pi_t(a) = 
\begin{cases}
    \arg\max_{a'}\,Q_t(a'), & \text{with probability }1-\epsilon,\\
    \text{randomly choose an arm}, & \text{with probability }\epsilon.
\end{cases}
$$

**(2) ChatGPT Prompt**

```text
請解釋 Epsilon-Greedy 演算法如何在多臂賭博機問題中平衡探索與利用，以及 ε 值（如 0.1、0.01）對學習曲線的影響。
```

**(3) Code \& Plot**

```python
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
```

![original image](https://cdn.mathpix.com/snip/images/yBvHKnD03m2oLQMz-R648Abalc0wjYmanG8B857j6uQ.original.fullsize.png)

> **Figure:** Epsilon-Greedy Average Cumulative Reward

**(4) Results Explanation**

- **Time complexity:** O(T)
- **Space complexity:** O(K)
- **Behavior:** Fixed ε ensures continued exploration but may slow convergence if ε too large; too small ε risks premature exploitation.

---

## 2. UCB (Upper Confidence Bound)

**(1) Algorithm Formula**

$$
a_t = \arg\max_a \bigl[Q_t(a) + c\sqrt{\tfrac{\ln t}{N_t(a)}}\bigr]
$$

**(2) ChatGPT Prompt**

```text
請說明 UCB 演算法中上信賴界項 c\sqrt{(\ln t)/N_t(a)} 的意義，及 c 值如何影響探索效率。
```

**(3) Code \& Plot**

```python
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
```

![original image](https://cdn.mathpix.com/snip/images/UxdmOQU3Ww1OFouD1PdIF0l2Bt7JdI7kEgz0z3BFMxA.original.fullsize.png)

> **Figure:** UCB Average Cumulative Reward

**(4) Results Explanation**

- **Time complexity:** O(KT)
- **Space complexity:** O(K)
- **Behavior:** Confidence bound term encourages exploring less-played arms; offers theoretical regret bound.

---

## 3. Softmax

**(1) Algorithm Formula**

$$
P_t(a) = \frac{\exp\bigl(Q_t(a)/\tau\bigr)}{\sum_{a'}\exp\bigl(Q_t(a')/\tau\bigr)}
$$

**(2) ChatGPT Prompt**

```text
請闡述 Softmax 演算法如何透過溫度參數 τ 調整探索與利用，以及 τ 大小對策略行為的影響。
```

**(3) Code \& Plot**

```python
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
```

![original image](https://cdn.mathpix.com/snip/images/z1cFPW2IVl0sJiBLTtDw1RPy08GbyzRzJatjUMmhuD8.original.fullsize.png)

> **Figure:** Softmax Average Cumulative Reward

**(4) Results Explanation**

- **Time complexity:** O(KT)
- **Space complexity:** O(K)
- **Behavior:** High τ → near-uniform exploration; low τ → greedy exploitation.

---

## 4. Thompson Sampling

**(1) Algorithm Formula**

$$
\theta_a \sim \mathrm{Beta}(S_a+1, F_a+1),\quad a_t = \arg\max_a \theta_a
$$

**(2) ChatGPT Prompt**

```text
請解釋 Thompson Sampling 如何利用貝式後驗分布（Beta 分布）自動平衡探索與利用，並討論其優勢。
```

**(3) Code \& Plot**

```python
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
```

![original image](https://cdn.mathpix.com/snip/images/SgQZVe2A_RcEQuPfYPog0f2yZmlJq7ZpLXsmD4Utqkg.original.fullsize.png)

> **Figure:** Thompson Sampling Average Cumulative Reward

**(4) Results Explanation**

- **Time complexity:** O(KT)
- **Space complexity:** O(K)
- **Behavior:** Adaptive sampling often converges fastest with low regret.

