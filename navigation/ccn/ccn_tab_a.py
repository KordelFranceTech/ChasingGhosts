"""
Adaptive tabular Q-learner with module-based growth / pruning / freezing.

Concepts:
- Each 'module' is like a column/layer. Q(s,a) = sum_over_modules Q_module(s,a)
- Growth: add a new module when reward instability or TD-instability indicates more capacity needed
- Pruning: remove modules whose mean abs-TD magnitude is below a threshold and visits are low
- Freezing: stop updating modules that are stable and have high contribution
"""

import numpy as np
import random
from collections import deque, defaultdict
import matplotlib.pyplot as plt


EPISODES: int = 1500
STEPS: int = 2000
MODULES: int = 10
# ------------------------------
# Environment (GasWorld)
# ------------------------------
class GasWorld:
    def __init__(self, size=20, source=(15, 15), diffusivity=0.005, wind=(0.01, 0.0), smell_range=5):
        self.size = size
        self.source = np.array(source, dtype=float)
        self.diffusivity = diffusivity
        self.wind = np.array(wind, dtype=float)
        self.smell_range = smell_range
        self.reset()

    def reset(self):
        self.agent_pos = np.array([random.randint(0, self.size-1), random.randint(0, self.size-1)], dtype=float)
        return self._state()

    def _state(self):
        return (int(self.agent_pos[0]), int(self.agent_pos[1]))

    def gas_concentration(self, pos):
        pos = np.array(pos, dtype=float)
        dist = np.linalg.norm(pos - self.source)
        wind_bias = np.dot(self.wind, (self.source - pos))
        concentration = np.exp(-(dist**2) * self.diffusivity + wind_bias)
        return float(concentration)

    def observation(self):
        # returns (smell_value, sight_flag)
        smell_val = self.gas_concentration(self.agent_pos)
        sight = (np.linalg.norm(self.agent_pos - self.source) <= self.smell_range)
        return smell_val, sight

    def step(self, action):
        # Actions: 0=up, 1=down, 2=left, 3=right
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        move = moves[action]
        new_pos = self.agent_pos + np.array(move)
        new_pos = np.clip(new_pos, 0, self.size - 1)
        self.agent_pos = new_pos
        concentration = self.gas_concentration(self.agent_pos)
        reward = float(concentration)
        done = (np.linalg.norm(self.agent_pos - self.source) <= 2.0)
        return self._state(), reward, done

# ------------------------------
# Adaptive Tabular Q-learner with modules
# ------------------------------
class ModuleQLearner:
    def __init__(self,
                 alpha=0.1, gamma=0.95, epsilon=0.1,
                 grow_window=50, reward_var_thresh=0.002, td_instability_thresh=0.01,
                 grow_patience=50, max_modules=MODULES,
                 prune_interval=50, prune_td_thresh=1e-4, prune_visit_thresh=50, prune_patience=50):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Modules: list of dicts mapping state -> 1D np.array (len=4 actions)
        self.modules = []            # list of q_tables (dicts)
        self.module_visits = []      # list of dicts mapping state -> np.array visit counts (ema)
        self.module_td_ema = []      # list of dicts mapping state -> np.array ema of abs TD updates
        self.module_frozen = []      # list of booleans

        # Start with one base module
        self._add_new_module()

        # Growth signals
        self.reward_window = deque(maxlen=grow_window)
        self.grow_window = grow_window
        self.reward_var_thresh = reward_var_thresh
        self.td_instability_thresh = td_instability_thresh
        self.grow_patience = grow_patience
        self.max_modules = max_modules
        self.no_growth_since = 0

        # Pruning
        self.prune_interval = prune_interval
        self.prune_td_thresh = prune_td_thresh
        self.prune_visit_thresh = prune_visit_thresh
        self.prune_patience = prune_patience
        self.no_prune_since = 0
        self.episode_count = 0

        # bookkeeping for plotting
        self.history = {
            'episode_reward': [],
            'num_modules': [],
            'module_sizes': []
        }

    def _add_new_module(self):
        self.modules.append(defaultdict(lambda: np.zeros(4, dtype=float)))
        self.module_visits.append(defaultdict(lambda: np.zeros(4, dtype=float)))
        self.module_td_ema.append(defaultdict(lambda: np.zeros(4, dtype=float)))
        self.module_frozen.append(False)
        print(f"[module] ADDED module #{len(self.modules)-1}")

    def _remove_module(self, idx):
        print(f"[module] REMOVING module #{idx}")
        del self.modules[idx]
        del self.module_visits[idx]
        del self.module_td_ema[idx]
        del self.module_frozen[idx]

    def get_effective_q(self, state):
        """Sum Q-values over modules for the given state."""
        qs = np.zeros(4, dtype=float)
        for m_idx, module in enumerate(self.modules):
            qs += module[state]  # defaultdict ensures zeros if missing
        return qs

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return int(np.argmax(self.get_effective_q(state)))

    def update(self, s, a, r, s_next, done):
        # compute target using summed Q
        q_next = 0.0 if done else np.max(self.get_effective_q(s_next))
        target = r + self.gamma * q_next

        # For each module, if not frozen, update Q_module(s,a) with own TD using target contribution
        # We compute module-level TD by considering each module's current q estimate and updating toward
        # the global target - note: this is a simple design choice for tabular modules.
        for m_idx, (module, frozen) in enumerate(zip(self.modules, self.module_frozen)):
            if frozen:
                continue
            q_sa = module[s][a]
            td_error = target - q_sa
            delta = self.alpha * td_error
            module[s][a] += delta

            # Update module visit EMA and TD EMA (abs)
            # EMA smoothing factor
            beta = 0.05
            self.module_visits[m_idx][s][a] = (1 - beta) * self.module_visits[m_idx][s][a] + beta * 1.0
            self.module_td_ema[m_idx][s][a] = (1 - beta) * self.module_td_ema[m_idx][s][a] + beta * abs(delta)

        # Keep reward window updated by caller (train_episode)

    def train_episode(self, env, max_steps=STEPS):
        s = env.reset()
        total_reward = 0.0
        for t in range(max_steps):
            a = self.choose_action(s)
            s_next, r, done = env.step(a)
            total_reward += r
            self.update(s, a, r, s_next, done)
            s = s_next
            if done:
                break

        # update bookkeeping
        self.episode_count += 1
        self.reward_window.append(total_reward)
        self.history['episode_reward'].append(total_reward)
        self.history['num_modules'].append(len(self.modules))
        self.history['module_sizes'].append([sum(np.sum(v) != 0 for v in mod.values()) for mod in self.modules])

        # Check growth trigger
        grew = False
        if len(self.reward_window) == self.grow_window:
            rew_var = np.var(self.reward_window)
            # compute mean module TD magnitude across states by sampling module_td_ema
            mean_td = 0.0
            count = 0
            for m_td in self.module_td_ema:
                for arr in m_td.values():
                    mean_td += np.mean(arr)
                    count += 1
            mean_td = mean_td / (count + 1e-9) if count > 0 else 0.0

            # growth condition: reward variance large OR mean_td large
            if (rew_var > self.reward_var_thresh) or (mean_td > self.td_instability_thresh):
                if len(self.modules) < self.max_modules:
                    self._add_new_module()
                    grew = True
                    self.no_growth_since = 0
                else:
                    # reached max modules, cannot grow
                    pass

        if not grew:
            self.no_growth_since += 1

        # Periodic pruning check
        pruned = False
        if self.episode_count % self.prune_interval == 0 and len(self.modules) > 1:
            # evaluate each non-base module for pruning eligibility
            removed_indices = []
            for m_idx in range(len(self.modules)-1, 0, -1):  # avoid removing base module 0
                td_mags = []
                visit_counts = []
                for s, arr in self.module_td_ema[m_idx].items():
                    td_mags.append(np.mean(arr))
                for s, arr in self.module_visits[m_idx].items():
                    visit_counts.append(np.mean(arr))
                mean_td = float(np.mean(td_mags)) if td_mags else 0.0
                mean_vis = float(np.mean(visit_counts)) if visit_counts else 0.0

                # prune condition: module is very stable (low TD), and low-visited
                if (mean_td < self.prune_td_thresh) and (mean_vis < self.prune_visit_thresh):
                    removed_indices.append(m_idx)

            if removed_indices:
                # remove all eligible modules
                for idx in sorted(removed_indices, reverse=True):
                    self._remove_module(idx)
                pruned = True

        if pruned:
            self.no_prune_since = 0
        else:
            self.no_prune_since += 1

        # Freezing: detect modules that are low TD mag but high contribution (high average Q)
        for m_idx in range(len(self.modules)):
            # compute mean abs TD and mean absolute Q contribution for this module
            td_mags = [np.mean(arr) for arr in self.module_td_ema[m_idx].values()] if self.module_td_ema[m_idx] else [0.0]
            mean_td = float(np.mean(td_mags))
            q_mags = [np.mean(np.abs(arr)) for arr in self.modules[m_idx].values()] if self.modules[m_idx] else [0.0]
            mean_q = float(np.mean(q_mags))
            # freeze condition: module stable (low TD) but contributes (mean_q above small threshold)
            if (mean_td < self.prune_td_thresh * 10) and (mean_q > 0.01):
                if not self.module_frozen[m_idx]:
                    self.module_frozen[m_idx] = True
                    print(f"[module] FREEZING module #{m_idx}")
            else:
                # allow unfreeze if instability returns
                if self.module_frozen[m_idx] and (mean_td > self.prune_td_thresh * 2):
                    self.module_frozen[m_idx] = False
                    print(f"[module] UNFREEZING module #{m_idx}")

        return total_reward

# ------------------------------
# Example experiment
# ------------------------------
def run_experiment(episodes=EPISODES, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    env = GasWorld(size=20, source=(15, 15), diffusivity=0.004, wind=(0.02, -0.005), smell_range=5)
    agent = ModuleQLearner(
        alpha=0.2, gamma=0.95, epsilon=0.15,
        grow_window=40, reward_var_thresh=0.02, td_instability_thresh=5e-3,
        grow_patience=60, max_modules=10,
        prune_interval=40, prune_td_thresh=2e-4, prune_visit_thresh=10, prune_patience=80
    )

    for ep in range(episodes):
        r = agent.train_episode(env)
        if (ep+1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes} reward={r:.3f} modules={len(agent.modules)}")

    # plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(agent.history['episode_reward'])
    axs[0].set_ylabel('Episode reward')
    axs[0].set_title('Adaptive Module Q-learning')

    axs[1].plot(agent.history['num_modules'], label='num_modules')
    axs[1].set_ylabel('Num modules')
    axs[1].set_xlabel('Episode')
    plt.tight_layout()
    plt.show()

    # Print final module stats
    print("Final num modules:", len(agent.modules))
    for i, mod in enumerate(agent.modules):
        nonzero_states = sum(1 for v in mod.values() if np.any(v != 0))
        print(f" Module {i}: nonzero-state-count = {nonzero_states}, frozen={agent.module_frozen[i]}")

    return agent, env

if __name__ == "__main__":
    agent, env = run_experiment(episodes=EPISODES, seed=42)

    # ------------------------------
    # Visualize path of best-performing agent
    # ------------------------------

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

    # Generate gas concentration field for visualization
    x = np.arange(env.size)
    y = np.arange(env.size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X, dtype=float)
    for i in range(env.size):
        for j in range(env.size):
            Z[i, j] = env.gas_concentration(np.array([i, j]))

    # Convert positions to arrays for plotting
    positions = np.array(positions)
    px, py = positions[:, 0], positions[:, 1]

    # Plot the path
    plt.figure(figsize=(6, 6))
    plt.imshow(Z.T, origin='lower', cmap='plasma', extent=[0, env.size, 0, env.size])
    plt.plot(px, py, color='white', linewidth=2, label='Agent path')
    plt.scatter(env.source[0], env.source[1], c='red', s=100, marker='*', label='Gas source')
    plt.scatter(px[0], py[0], c='green', s=50, label='Start')
    plt.title(f"Path of Agent")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend()
    plt.show()

