import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

EPISODES: int = 1000
STEPS: int = 500

# ------------------------------
# Environment
# ------------------------------
class GasWorld:
    def __init__(self, size=20, source=(15, 15), diffusivity=0.02, wind=(0.01, 0.0)):
        self.size = size
        self.source = np.array(source, dtype=float)
        self.diffusivity = diffusivity
        self.wind = np.array(wind, dtype=float)
        self.reset()

    def reset(self):
        self.agent_pos = np.array([random.randint(0, self.size - 1), random.randint(0, self.size - 1)], dtype=float)
        return tuple(self.agent_pos.astype(int))

    def gas_concentration(self, pos):
        dist = np.linalg.norm(pos - self.source)
        wind_bias = np.dot(self.wind, (self.source - pos))
        concentration = np.exp(-(dist**2) * self.diffusivity + wind_bias)
        return concentration

    def step(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        move = moves[action]
        new_pos = self.agent_pos + np.array(move)
        new_pos = np.clip(new_pos, 0, self.size - 1)
        self.agent_pos = new_pos

        concentration = self.gas_concentration(self.agent_pos)
        reward = concentration
        done = np.linalg.norm(self.agent_pos - self.source) <= 2.0
        return tuple(self.agent_pos.astype(int)), reward, done

# ------------------------------
# Adaptive Q-learning agent
# ------------------------------
class AdaptiveQLearner:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1,
                 growth_threshold=0.02, prune_threshold=0.005,
                 variance_window=50):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.growth_threshold = growth_threshold
        self.prune_threshold = prune_threshold
        self.q_table = {}
        self.visits = {}
        self.recent_rewards = deque(maxlen=variance_window)
        self.variance_window = variance_window
        self.q_update_variance = deque(maxlen=variance_window)
        self.grow_events = 0
        self.prune_events = 0

    def get_qs(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
            self.visits[state] = np.zeros(4)
            self.grow_events += 1
        return self.q_table[state]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.get_qs(state))

    def update(self, s, a, r, s_next, done):
        self.get_qs(s)
        old_q = self.q_table[s][a]
        q_next = np.max(self.get_qs(s_next)) if not done else 0.0
        target = r + self.gamma * q_next
        new_q = old_q + self.alpha * (target - old_q)
        td_update = abs(new_q - old_q)
        self.q_table[s][a] = new_q
        self.visits[s][a] += 1
        self.q_update_variance.append(td_update)
        return td_update

    def growth_condition(self):
        # Reward variance threshold: if recent reward variance > growth_threshold → grow
        if len(self.recent_rewards) < self.variance_window:
            return False
        reward_var = np.var(self.recent_rewards)
        return reward_var > self.growth_threshold

    def prune_condition(self):
        # If mean TD update magnitude < prune_threshold → prune
        if len(self.q_update_variance) < self.variance_window:
            return False
        mean_td = np.mean(self.q_update_variance)
        return mean_td < self.prune_threshold

    def prune(self):
        to_delete = []
        for s in list(self.q_table.keys()):
            if np.all(self.visits[s] < 2):  # unvisited states
                to_delete.append(s)
        for s in to_delete:
            del self.q_table[s]
            del self.visits[s]
            self.prune_events += 1

    def train_episode(self, env, max_steps=STEPS):
        s = env.reset()
        total_reward = 0
        for _ in range(max_steps):
            a = self.choose_action(s)
            s_next, r, done = env.step(a)
            td = self.update(s, a, r, s_next, done)
            total_reward += r
            s = s_next
            if done:
                break
        self.recent_rewards.append(total_reward)

        if self.prune_condition():
            self.prune()

        return total_reward


# ------------------------------
# Experiment Sweep
# ------------------------------
def run_experiment(growth_threshold, prune_threshold, episodes=EPISODES):
    env = GasWorld(size=20, source=(15, 15))
    agent = AdaptiveQLearner(
        growth_threshold=growth_threshold,
        prune_threshold=prune_threshold
    )

    rewards = []
    for _ in range(episodes):
        r = agent.train_episode(env)
        rewards.append(r)

    mean_reward = np.mean(rewards[-50:])
    final_size = len(agent.q_table)
    return mean_reward, final_size, agent.grow_events, agent.prune_events


# Sweep parameters
growth_values = np.linspace(0.005, 0.05, 5)
prune_values = np.linspace(0.001, 0.02, 5)

results_mean = np.zeros((len(growth_values), len(prune_values)))
results_size = np.zeros_like(results_mean)
results_grow_events = np.zeros_like(results_mean)
results_prune_events = np.zeros_like(results_mean)

for i, gth in enumerate(growth_values):
    for j, pth in enumerate(prune_values):
        mean_r, final_size, grows, prunes = run_experiment(gth, pth)
        results_mean[i, j] = mean_r
        results_size[i, j] = final_size
        results_grow_events[i, j] = grows
        results_prune_events[i, j] = prunes
        print(f"Growth {gth:.3f}, Prune {pth:.3f} -> Reward {mean_r:.3f}, Size {final_size}, Grows {grows}, Prunes {prunes}")

# ------------------------------
# Visualization
# ------------------------------
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
xlabels = [f"{p:.3f}" for p in prune_values]
ylabels = [f"{g:.3f}" for g in growth_values]

im1 = axs[0].imshow(results_mean, cmap='viridis')
axs[0].set_title("Mean Reward")
axs[0].set_xticks(range(len(prune_values)))
axs[0].set_yticks(range(len(growth_values)))
axs[0].set_xticklabels(xlabels, rotation=45)
axs[0].set_yticklabels(ylabels)
plt.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(results_size, cmap='plasma')
axs[1].set_title("Final Q-table Size")
axs[1].set_xticks(range(len(prune_values)))
axs[1].set_yticks(range(len(growth_values)))
axs[1].set_xticklabels(xlabels, rotation=45)
axs[1].set_yticklabels(ylabels)
plt.colorbar(im2, ax=axs[1])

im3 = axs[2].imshow(results_prune_events, cmap='cool')
axs[2].set_title("Prune Events")
axs[2].set_xticks(range(len(prune_values)))
axs[2].set_yticks(range(len(growth_values)))
axs[2].set_xticklabels(xlabels, rotation=45)
axs[2].set_yticklabels(ylabels)
plt.colorbar(im3, ax=axs[2])

plt.suptitle("Sensitivity to Growth and Pruning Thresholds")
plt.tight_layout()
plt.show()


# ------------------------------
# Visualize path of best-performing agent
# ------------------------------

# Find best thresholds from the sweep
best_idx = np.unravel_index(np.argmax(results_mean), results_mean.shape)
best_growth = growth_values[best_idx[0]]
best_prune = prune_values[best_idx[1]]

print(f"\nBest-performing thresholds:")
print(f"Growth threshold = {best_growth:.4f}, Prune threshold = {best_prune:.4f}")
print(f"Mean reward = {results_mean[best_idx]:.4f}")


# --- Reuse best-performing agent and environment ---
env = GasWorld(size=20, source=(15, 15))
agent = AdaptiveQLearner(
    growth_threshold=best_growth,
    prune_threshold=best_prune
)

positions = []
s = env.reset()
positions.append(s)

done = False
steps = 0
while not done and steps < STEPS:
    a = agent.choose_action(s)
    s_next, r, done = env.step(a)
    agent.update(s, a, r, s_next, done)
    positions.append(s_next)
    s = s_next
    steps += 1

positions = np.array(positions)

# Generate gas concentration field
x = np.arange(env.size)
y = np.arange(env.size)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X, dtype=float)
for i in range(env.size):
    for j in range(env.size):
        Z[i, j] = env.gas_concentration(np.array([i, j]))

# --- Set up animation ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, env.size)
ax.set_ylim(0, env.size)
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_title(f"Agent Path Animation\n(growth={best_growth:.4f}, prune={best_prune:.4f})")

# Show gas concentration as background
im = ax.imshow(Z.T, origin='lower', cmap='plasma', extent=[0, env.size, 0, env.size])

# Plot source
ax.scatter(env.source[0], env.source[1], c='red', s=100, marker='*', label='Gas source')
# Plot start
ax.scatter(positions[0,0], positions[0,1], c='green', s=50, label='Start')

# Agent marker
agent_dot, = ax.plot([], [], 'wo', markersize=8, label='Agent')

# Path line
path_line, = ax.plot([], [], 'w-', linewidth=2)

ax.legend(loc='upper left')

def init():
    agent_dot.set_data([], [])
    path_line.set_data([], [])
    return agent_dot, path_line

def animate(i):
    agent_dot.set_data(positions[i,0], positions[i,1])
    path_line.set_data(positions[:i+1,0], positions[:i+1,1])
    return agent_dot, path_line

ani = animation.FuncAnimation(
    fig, animate, frames=len(positions), init_func=init,
    interval=200, blit=True, repeat=False
)

plt.show()

