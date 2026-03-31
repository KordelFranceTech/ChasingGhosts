import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ------------------------------
# Environment
# ------------------------------
class LightweightPlumEnv:
    def __init__(self, size=20, source=(15, 15), diffusivity=0.05, wind=(0.01, 0.0)):
        self.size = size
        self.source = np.array(source, dtype=float)
        self.diffusivity = diffusivity
        self.wind = np.array(wind, dtype=float)
        self.reset()

    def reset(self):
        self.agent_pos = np.array([random.randint(0, self.size-1), random.randint(0, self.size-1)], dtype=float)
        return self._state()

    def _state(self):
        # normalized position [0,1] for NN input
        return np.array(self.agent_pos / self.size, dtype=np.float32)

    def gas_concentration(self, pos):
        pos = np.array(pos, dtype=float)
        dist = np.linalg.norm(pos - self.source)
        wind_bias = np.dot(self.wind, (self.source - pos))
        concentration = np.exp(-(dist**2) * self.diffusivity + wind_bias)
        return float(concentration)

    def step(self, action):
        moves = {0: (-1,0),1:(1,0),2:(0,-1),3:(0,1)}
        self.agent_pos += np.array(moves[action])
        self.agent_pos = np.clip(self.agent_pos, 0, self.size-1)
        reward = self.gas_concentration(self.agent_pos)
        done = np.linalg.norm(self.agent_pos - self.source) <= 2.0
        return self._state(), reward, done

# ------------------------------
# Neural Module Q-Learner
# ------------------------------
class NeuralModuleQLearner:
    def __init__(self,
                 input_dim=2,
                 hidden_dim=32,
                 output_dim=4,
                 alpha=0.001,
                 gamma=0.95,
                 epsilon=0.1,
                 grow_window=50,
                 reward_var_thresh=0.002,
                 td_instability_thresh=0.01,
                 max_layers=5,
                 prune_interval=50,
                 prune_threshold=0.01,
                 prune_patience=50,
                 device='cpu'):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.device = device

        # list of layers (modules)
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim).to(device)])
        self.output_layer = nn.Linear(hidden_dim, output_dim).to(device)
        self.optim = optim.Adam(self.parameters(), lr=alpha)

        # bookkeeping
        self.reward_window = deque(maxlen=grow_window)
        self.grow_window = grow_window
        self.reward_var_thresh = reward_var_thresh
        self.td_instability_thresh = td_instability_thresh
        self.max_layers = max_layers

        self.prune_interval = prune_interval
        self.prune_threshold = prune_threshold
        self.prune_patience = prune_patience

        # track mean absolute TD per step
        self.td_window = deque(maxlen=grow_window)
        self.episode_count = 0

    def parameters(self):
        params = list(self.layers.parameters()) + list(self.output_layer.parameters())
        return params

    def forward(self, x):
        # x: tensor shape [batch, input_dim]
        out = x
        for layer in self.layers:
            out = F.relu(layer(out))
        q = self.output_layer(out)
        return q

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_dim-1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.forward(state_tensor)
        return int(torch.argmax(q_values).item())

    def update(self, state, action, reward, next_state, done):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward_t = torch.tensor([reward], dtype=torch.float32).to(self.device)

        # compute target
        with torch.no_grad():
            q_next = self.forward(next_state_t)
            q_next_max = torch.max(q_next)
            target = reward_t + (0.0 if done else self.gamma * q_next_max)

        q_pred = self.forward(state_t)[0, action]
        td_error = target - q_pred
        loss = td_error.pow(2)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # track TD magnitude
        self.td_window.append(abs(td_error.item()))
        return td_error.item()

    def train_episode(self, env, max_steps=200):
        state = env.reset()
        total_reward = 0
        for _ in range(max_steps):
            action = self.choose_action(state)
            next_state, reward, done = env.step(action)
            self.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break

        self.reward_window.append(total_reward)
        self.episode_count += 1

        # check growth
        if len(self.reward_window) == self.grow_window:
            reward_var = np.var(self.reward_window)
            td_mean = np.mean(self.td_window) if self.td_window else 0.0
            if (reward_var > self.reward_var_thresh or td_mean > self.td_instability_thresh) and len(self.layers) < self.max_layers:
                self.add_layer()

        # check pruning
        if self.episode_count % self.prune_interval == 0:
            self.prune_neurons()

        return total_reward

    def add_layer(self):
        print(f"[module] Adding new layer")
        new_layer = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
        self.layers.append(new_layer)
        self.optim = optim.Adam(self.parameters(), lr=self.alpha)

    def prune_neurons(self):
        # Simple pruning: remove neurons with small absolute weights in hidden layers
        for i, layer in enumerate(self.layers):
            with torch.no_grad():
                weight_abs = torch.abs(layer.weight).sum(dim=1)  # sum over input connections
                mask = weight_abs > self.prune_threshold
                if mask.sum() < layer.weight.size(0):
                    # prune neurons
                    keep_idx = torch.nonzero(mask).flatten()
                    layer.weight.data = layer.weight.data[keep_idx,:]
                    layer.bias.data = layer.bias.data[keep_idx]
                    self.hidden_dim = layer.weight.size(0)
                    print(f"[prune] Layer {i} pruned to {self.hidden_dim} neurons")
        # rebuild output layer to match last layer
        last_hidden = self.layers[-1].weight.size(0)
        self.output_layer = nn.Linear(last_hidden, self.output_dim).to(self.device)
        self.optim = optim.Adam(self.parameters(), lr=self.alpha)

# ------------------------------
# Experiment Example
# ------------------------------
def run_neural_experiment(episodes=300):
    env = LightweightPlumEnv(size=20, source=(15,15))
    agent = NeuralModuleQLearner(
        input_dim=2, hidden_dim=16, output_dim=4,
        alpha=0.005, gamma=0.95, epsilon=0.1,
        grow_window=40, reward_var_thresh=0.02, td_instability_thresh=0.01,
        max_layers=3, prune_interval=40, prune_threshold=0.01
    )

    rewards = []
    for ep in range(episodes):
        r = agent.train_episode(env)
        rewards.append(r)
        if (ep+1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes} reward={r:.3f}, layers={len(agent.layers)}")

    # Plot rewards
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Neural Adaptive Q-learning")
    plt.show()

    return agent, env


# --- Record neural paths with layer size tracking ---
def record_neural_path_with_layers(agent, env, max_steps=200):
    state = env.reset()
    positions = [state * env.size]
    layer_sizes = [sum(layer.weight.size(0) for layer in agent.layers)]  # total neurons

    done = False
    steps = 0
    while not done and steps < max_steps:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        positions.append(next_state * env.size)
        # record current total neurons
        layer_sizes.append(sum(layer.weight.size(0) for layer in agent.layers))
        state = next_state
        steps += 1

    positions = np.array(positions)
    layer_sizes = np.array(layer_sizes)
    return positions, layer_sizes


if __name__ == "__main__":

    # --- Sweep parameters ---
    growth_values = [0.01, 0.02, 0.03, 0.07]  # reward variance threshold for growth
    prune_values = [0.002, 0.005, 0.01, 0.02]  # pruning threshold for neurons
    episodes_per_run = 2000
    max_steps = 300

    # Store results
    results_mean_reward = np.zeros((len(growth_values), len(prune_values)))
    results_layers = np.zeros_like(results_mean_reward)
    paths_dict = {}  # to store paths for animation

    # --- Sweep loop with layer size tracking ---
    paths_layers_dict = {}  # (positions, layer_sizes)
    for i, gth in enumerate(growth_values):
        for j, pth in enumerate(prune_values):
            print(f"Running growth={gth}, prune={pth}")
            env = LightweightPlumEnv(size=20, source=(15,15))
            agent = NeuralModuleQLearner(
                input_dim=2, hidden_dim=16, output_dim=4,
                alpha=0.005, gamma=0.95, epsilon=0.1,
                grow_window=40, reward_var_thresh=gth,
                td_instability_thresh=0.01,
                max_layers=3, prune_interval=40,
                prune_threshold=pth
            )
            # Train the agent
            for ep in range(episodes_per_run):
                agent.train_episode(env, max_steps)

            # Record path and layer sizes for last episode
            positions, layer_sizes = record_neural_path_with_layers(agent, env, max_steps)
            paths_layers_dict[(i,j)] = (positions, layer_sizes)

    # --- Animate paths with dynamic layer sizes ---
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Neural Agent Paths with Dynamic Layer Size")

    # Background gas field
    X, Y = np.meshgrid(range(env.size), range(env.size))
    Z = np.zeros_like(X, dtype=float)
    for i in range(env.size):
        for j in range(env.size):
            Z[i,j] = env.gas_concentration(np.array([i,j]))
    ax.imshow(Z.T, origin='lower', cmap='plasma', extent=[0, env.size, 0, env.size])

    # Gas source
    ax.scatter(env.source[0], env.source[1], c='red', s=100, marker='*', label='Source')

    # Lines and dots
    lines = []
    dots = []
    text_layers = []
    for key, (path, layer_sizes) in paths_layers_dict.items():
        line, = ax.plot([], [], lw=2, label=f"g={growth_values[key[0]]},p={prune_values[key[1]]}")
        dot, = ax.plot([], [], 'wo', markersize=5)
        text = ax.text(0.02, 0.95 - 0.05*len(text_layers), '', transform=ax.transAxes, color='white', fontsize=8)
        lines.append(line)
        dots.append(dot)
        text_layers.append(text)

    ax.legend(loc='upper left', fontsize=8)

    def init():
        for line,dot,text in zip(lines,dots,text_layers):
            line.set_data([],[])
            dot.set_data([],[])
            text.set_text('')
        return lines + dots + text_layers

    def animate(frame):
        for idx, (line,dot,text) in enumerate(zip(lines,dots,text_layers)):
            path, layer_sizes = list(paths_layers_dict.values())[idx]
            max_frame = len(path)
            current_frame = min(frame, max_frame-1)
            line.set_data(path[:current_frame+1,0], path[:current_frame+1,1])
            dot.set_data(path[current_frame,0], path[current_frame,1])
            text.set_text(f"Layers={layer_sizes[current_frame]}")
        return lines + dots + text_layers

    max_frames = max([len(p[0]) for p in paths_layers_dict.values()])
    ani = animation.FuncAnimation(fig, animate, frames=max_frames, init_func=init, interval=200, blit=True, repeat=False)
    plt.show()
