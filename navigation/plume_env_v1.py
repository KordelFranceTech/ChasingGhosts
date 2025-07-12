# Full Simulation: Olfactory Navigation with Partial Observability and Visualization

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# ----------------------------- ENVIRONMENT -----------------------------

class OlfactoryGridWorld(gym.Env):
    def __init__(self, grid_size=8):
        super(OlfactoryGridWorld, self).__init__()
        self.grid_size = grid_size
        self.start_pos = (0, 0)
        self.source_pos = (grid_size - 1, grid_size - 1)

        # Observation: [x_norm, y_norm] + 3x3 scent + 3x3 object = 2 + 9 + 9 = 20
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(20,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)  # up, right, down, left

        self.reset()

    def reset(self):
        self.agent_pos = list(self.start_pos)
        self.object_map = self.generate_object_confidence_map()
        self.wind_direction = np.random.choice(['N', 'S', 'E', 'W'])
        self.scent_map = self.generate_scent_map_with_wind()
        self.trajectory = [tuple(self.agent_pos)]
        return self.get_observation()

    def step(self, action):
        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[1] < self.grid_size - 1:
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] < self.grid_size - 1:
            self.agent_pos[0] += 1
        elif action == 3 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1

        self.trajectory.append(tuple(self.agent_pos))
        obs = self.get_observation()
        reward = self.scent_map[tuple(self.agent_pos)] + self.object_map[tuple(self.agent_pos)]
        done = (tuple(self.agent_pos) == self.source_pos)
        return obs, reward, done, {}

    def get_observation(self):
        x, y = self.agent_pos
        scent_patch = self.scent_map[max(0, x - 1):x + 2, max(0, y - 1):y + 2]
        object_patch = self.object_map[max(0, x - 1):x + 2, max(0, y - 1):y + 2]
        scent_patch = np.pad(scent_patch, ((0, 3 - scent_patch.shape[0]), (0, 3 - scent_patch.shape[1])), constant_values=0)
        object_patch = np.pad(object_patch, ((0, 3 - object_patch.shape[0]), (0, 3 - object_patch.shape[1])), constant_values=0)
        return np.concatenate([
            [x / (self.grid_size - 1), y / (self.grid_size - 1)],
            scent_patch.flatten(),
            object_patch.flatten()
        ]).astype(np.float32)

    def generate_scent_map_with_wind(self):
        scent_map = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dx, dy = i - self.source_pos[0], j - self.source_pos[1]
                dist = np.sqrt(dx ** 2 + dy ** 2)
                if self.wind_direction == 'N':
                    wind_factor = np.exp(-0.5 * (dx - 2) ** 2)
                elif self.wind_direction == 'S':
                    wind_factor = np.exp(-0.5 * (dx + 2) ** 2)
                elif self.wind_direction == 'E':
                    wind_factor = np.exp(-0.5 * (dy - 2) ** 2)
                else:  # 'W'
                    wind_factor = np.exp(-0.5 * (dy + 2) ** 2)
                scent_map[i, j] = np.exp(-dist) * wind_factor
        return scent_map

    def generate_object_confidence_map(self):
        np.random.seed(42)
        conf = np.random.rand(self.grid_size, self.grid_size) * 0.2
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                x, y = self.source_pos[0] + dx, self.source_pos[1] + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    conf[x, y] = 0.8 + 0.2 * np.random.rand()
        return conf

# ----------------------------- DQN AGENT -----------------------------

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, trace_decay=0.9):
        self.gamma = gamma
        self.trace_decay = trace_decay
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(10000)

        self.eligibility_traces = [torch.zeros_like(p) for p in self.policy_net.parameters()]

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        with torch.no_grad():
            return self.policy_net(torch.FloatTensor(state)).argmax().item()

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = Transition(*zip(*self.memory.sample(batch_size)))
        states = torch.FloatTensor(batch.state)
        actions = torch.LongTensor(batch.action).unsqueeze(1)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1)
        next_states = torch.FloatTensor(batch.next_state)
        dones = torch.FloatTensor(batch.done).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + (1 - dones) * self.gamma * max_next_q

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()

        for i, param in enumerate(self.policy_net.parameters()):
            self.eligibility_traces[i] = self.gamma * self.trace_decay * self.eligibility_traces[i] + param.grad
            param.grad = self.eligibility_traces[i]

        self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ----------------------------- VISUALIZATION -----------------------------

def render_env(env, agent_pos=None, scent_map=None, object_map=None):
    grid_size = env.grid_size
    fig, ax = plt.subplots(figsize=(6, 6))

    for x in range(grid_size):
        for y in range(grid_size):
            rect = patches.Rectangle((y, grid_size - 1 - x), 1, 1, linewidth=1, edgecolor='black', facecolor='white')
            ax.add_patch(rect)

            if scent_map is not None:
                scent_strength = scent_map[x, y]
                color = (1.0, 1.0 - scent_strength, 1.0 - scent_strength)
                rect.set_facecolor(color)

            if object_map is not None and object_map[x, y] > 0.6:
                ax.text(y + 0.5, grid_size - 1 - x + 0.5, '💣', ha='center', va='center', fontsize=12)

    if agent_pos:
        ax.add_patch(patches.Circle((agent_pos[1] + 0.5, grid_size - 1 - agent_pos[0] + 0.5), 0.3, color='blue'))

    ax.add_patch(patches.Circle((env.source_pos[1] + 0.5, grid_size - 1 - env.source_pos[0] + 0.5), 0.3, color='red'))

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.grid(True)
    plt.show()

# ----------------------------- TRAINING LOOP -----------------------------

if __name__ == '__main__':
    env = OlfactoryGridWorld(grid_size=8)
    agent = DQNAgent(state_dim=20, action_dim=4)

    episodes = 200
    batch_size = 32

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count: int = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, next_state, reward, done)
            state = next_state
            total_reward += reward
            agent.update(batch_size)
            # Visualize every 20 episodes
            # if step_count % 20 == 0:
            if ep == 0:
                render_env(env, agent_pos=env.agent_pos, scent_map=env.scent_map, object_map=env.object_map)
            step_count += 1
        agent.update_target_network()
        print(f"Episode {ep} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f}")

