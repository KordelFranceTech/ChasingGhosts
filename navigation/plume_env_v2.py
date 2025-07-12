# Full 3D Olfactory GridWorld Simulation with DQN and Visualization

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from collections import deque, namedtuple

# ----------------------------- ENVIRONMENT -----------------------------

class OlfactoryGridWorld3D(gym.Env):
    def __init__(self, grid_size=(8, 8, 5), start_pos=(0, 0, 0), source_pos=(7, 7, 4)):
        super(OlfactoryGridWorld3D, self).__init__()
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.source_pos = source_pos

        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(57,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.agent_pos = list(self.start_pos)
        self.object_map = self.generate_object_confidence_map()
        self.scent_map = self.generate_3d_scent_map()
        self.trajectory = [tuple(self.agent_pos)]
        return self.get_observation()

    def step(self, action):
        x, y, z = self.agent_pos
        if action == 0 and x > 0: x -= 1
        elif action == 1 and x < self.grid_size[0] - 1: x += 1
        elif action == 2 and y < self.grid_size[1] - 1: y += 1
        elif action == 3 and y > 0: y -= 1
        elif action == 4 and z < self.grid_size[2] - 1: z += 1
        elif action == 5 and z > 0: z -= 1
        self.agent_pos = [x, y, z]
        self.trajectory.append(tuple(self.agent_pos))
        reward = self.scent_map[x, y, z] + self.object_map[x, y, z]
        done = (tuple(self.agent_pos) == self.source_pos)
        return self.get_observation(), reward, done, {}

    def get_observation(self):
        x, y, z = self.agent_pos
        obs = [x / (self.grid_size[0] - 1), y / (self.grid_size[1] - 1), z / (self.grid_size[2] - 1)]
        scent_patch = self.get_local_patch(self.scent_map, 3)
        object_patch = self.get_local_patch(self.object_map, 3)
        return np.concatenate([obs, scent_patch.flatten(), object_patch.flatten()]).astype(np.float32)

    def get_local_patch(self, volume, size):
        x, y, z = self.agent_pos
        pad_width = size // 2
        padded = np.pad(volume, pad_width, mode='constant', constant_values=0)
        x_p, y_p, z_p = x + pad_width, y + pad_width, z + pad_width
        return padded[x_p - 1:x_p + 2, y_p - 1:y_p + 2, z_p - 1:z_p + 2]

    def generate_3d_scent_map(self):
        scent_map = np.zeros(self.grid_size)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for z in range(self.grid_size[2]):
                    dist = np.linalg.norm(np.array([x, y, z]) - np.array(self.source_pos))
                    scent_map[x, y, z] = np.exp(-dist)
        return scent_map

    def generate_object_confidence_map(self):
        conf = np.random.rand(*self.grid_size) * 0.2
        sx, sy, sz = self.source_pos
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    x, y, z = sx + dx, sy + dy, sz + dz
                    if (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] and 0 <= z < self.grid_size[2]):
                        conf[x, y, z] = 0.8 + 0.2 * np.random.rand()
        return conf

# ----------------------------- VISUALIZATION -----------------------------

def plot_3d_trajectory(env):
    traj = np.array(env.trajectory)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sx, sy, sz = env.source_pos
    ax.scatter(sy, sx, sz, color='red', s=100, label='Source')
    ax.plot(traj[:,1], traj[:,0], traj[:,2], marker='o', color='blue', label='Agent Trajectory')
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Z')
    ax.set_title('3D Agent Trajectory in Olfactory GridWorld')
    ax.legend()
    plt.show()

def animate_3d_trajectory(env):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    traj = np.array(env.trajectory)

    def update(frame):
        ax.clear()
        ax.set_xlim(0, env.grid_size[1])
        ax.set_ylim(0, env.grid_size[0])
        ax.set_zlim(0, env.grid_size[2])
        ax.set_xlabel("Y")
        ax.set_ylabel("X")
        ax.set_zlabel("Z")
        sx, sy, sz = env.source_pos
        ax.scatter(sy, sx, sz, color='red', s=100, label='Source')
        ax.plot(traj[:frame+1,1], traj[:frame+1,0], traj[:frame+1,2], marker='o', color='blue')
        ax.set_title(f"Step {frame+1}/{len(traj)}")
        ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=len(traj), interval=400)
    ani.save("3d_trajectory_animation.mp4", writer="ffmpeg")
    plt.show()


def animate_scent_plume(env, wind_vector=(1, 1, 0.2), steps=40, upsample_factor=3,
                        decay_rate=0.01, diffusion_sigma=1.0, save_path="scent_plume.gif"):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from scipy.ndimage import zoom, gaussian_filter, shift
    from matplotlib import cm
    from matplotlib.colors import Normalize
    import numpy as np

    scent = env.scent_map.copy()
    source = np.array(env.source_pos)
    scaled_source = source * upsample_factor

    # Upsample the original scent volume
    scent_upsampled = zoom(scent, zoom=upsample_factor, order=1)

    # Setup figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        nonlocal scent_upsampled
        ax.clear()

        # Step 1: Apply decay
        scent_upsampled *= (1 - decay_rate)

        # Step 2: Diffusion (Gaussian blur)
        scent_upsampled = gaussian_filter(scent_upsampled, sigma=diffusion_sigma)

        # Step 3: Wind transport (shift)
        scent_upsampled = shift(scent_upsampled, shift=[-v for v in wind_vector], order=1, mode='constant', cval=0.0)

        # Step 4: Extract scent above threshold
        threshold = 0.01
        x, y, z = np.where(scent_upsampled > threshold)
        coords = np.stack([x, y, z], axis=1)
        values = scent_upsampled[x, y, z]

        # Step 5: Distance-based color
        dists = np.linalg.norm(coords - scaled_source, axis=1)
        norm = Normalize(vmin=0, vmax=50)
        colors = cm.plasma(norm(dists))

        ax.scatter(y, x, z, c=colors, alpha=0.2, s=1)

        # Wind vector field
        step = 6
        xg, yg, zg = np.meshgrid(
            np.arange(0, scent_upsampled.shape[0], step),
            np.arange(0, scent_upsampled.shape[1], step),
            np.arange(0, scent_upsampled.shape[2], step),
            indexing='ij'
        )
        u = np.full_like(xg, wind_vector[0])
        v = np.full_like(yg, wind_vector[1])
        w = np.full_like(zg, wind_vector[2])
        ax.quiver(yg, xg, zg, v, u, w, length=2, normalize=True, color='gray', alpha=0.3)

        # Plot source
        ax.scatter(scaled_source[1], scaled_source[0], scaled_source[2], color='red', s=100, label='Source')

        ax.set_xlim(0, scent_upsampled.shape[1])
        ax.set_ylim(0, scent_upsampled.shape[0])
        ax.set_zlim(0, scent_upsampled.shape[2])
        ax.set_xlabel("Y")
        ax.set_ylabel("X")
        ax.set_zlabel("Z")
        ax.set_title(f"Scent Plume Dispersion & Decay (Step {frame+1})")
        ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=300)
    ani.save(save_path, writer='pillow', fps=5)  # save as GIF
    plt.close(fig)



def plot_scent_heatmap_voxels(env):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scent = env.scent_map
    threshold = 0.01
    x, y, z = np.where(scent > threshold)
    colors = plt.cm.viridis(scent[x, y, z])
    ax.scatter(y, x, z, c=colors, marker='s', alpha=0.6)
    ax.set_xlabel("Y")
    ax.set_ylabel("X")
    ax.set_zlabel("Z")
    ax.set_title("Scent Intensity Voxel Visualization")
    plt.show()

def plot_scent_heatmap(env, wind_vector=(1, 1, 0.2)):
    from scipy.ndimage import zoom
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.colors import Normalize

    scent = env.scent_map
    source = np.array(env.source_pos)

    # Upsample the scent map to make it smoother
    upsample_factor = 3
    upsampled = zoom(scent, zoom=upsample_factor, order=1)
    threshold = 0.02
    x, y, z = np.where(upsampled > threshold)

    coords = np.stack([x, y, z], axis=1)
    values = upsampled[x, y, z]

    # Distance-based coloring (proximity to source)
    scaled_source = source * upsample_factor
    dists = np.linalg.norm(coords - scaled_source, axis=1)
    norm = Normalize(vmin=dists.min(), vmax=dists.max())
    colors = cm.plasma(norm(dists))  # or viridis, inferno, magma, etc.

    # Plot setup
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(y, x, z, c=colors, alpha=0.2, s=5)

    # Draw the source
    ax.scatter(scaled_source[1], scaled_source[0], scaled_source[2],
               color='red', s=100, label='Source')

    # Wind vector field: draw arrows showing wind direction at intervals
    step = 4
    xg, yg, zg = np.meshgrid(
        np.arange(0, upsampled.shape[0], step),
        np.arange(0, upsampled.shape[1], step),
        np.arange(0, upsampled.shape[2], step),
        indexing='ij'
    )
    u = np.full_like(xg, wind_vector[0])
    v = np.full_like(yg, wind_vector[1])
    w = np.full_like(zg, wind_vector[2])
    ax.quiver(yg, xg, zg, v, u, w, length=2, normalize=True, color='gray', alpha=0.4)

    ax.set_xlabel("Y")
    ax.set_ylabel("X")
    ax.set_zlabel("Z")
    ax.set_title("Scent Plume with Wind Flow and Proximity Coloring")
    ax.legend()
    plt.show()



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
    def __init__(self, state_dim, action_dim, memory, gamma=0.99, lr=1e-4, trace_decay=0.9):
        self.gamma = gamma
        self.trace_decay = trace_decay
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.memory = memory
        self.eligibility_traces = [torch.zeros_like(p) for p in self.policy_net.parameters()]

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 5)
        with torch.no_grad():
            return self.policy_net(torch.FloatTensor(state)).argmax().item()

    # Q learning
    def update_dqn(self, batch_size):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

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

    # Expected SARSA
    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(batch.state)
        actions = torch.LongTensor(batch.action).unsqueeze(1)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1)
        next_states = torch.FloatTensor(batch.next_state)
        dones = torch.FloatTensor(batch.done).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states)  # shape: [batch_size, num_actions]
            num_actions = next_q_values.shape[1]
            greedy_actions = next_q_values.argmax(dim=1, keepdim=True)

            probs = torch.full_like(next_q_values, self.epsilon / num_actions)
            probs.scatter_(1, greedy_actions, 1 - self.epsilon + (self.epsilon / num_actions))

            expected_next_q = (next_q_values * probs).sum(dim=1, keepdim=True)
            target = rewards + (1 - dones) * self.gamma * expected_next_q

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

# ----------------------------- TRAINING LOOP -----------------------------

if __name__ == '__main__':
    env = OlfactoryGridWorld3D()
    memory = ReplayMemory(10000)
    agent = DQNAgent(state_dim=57, action_dim=6, memory=memory)

    episodes = 50
    batch_size = 32
    rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            memory.push(state, action, next_state, reward, done)
            agent.update(batch_size)
            state = next_state
            total_reward += reward

        agent.update_target_network()
        rewards.append(total_reward)
        print(f"Episode {ep}: Total Reward = {total_reward:.2f}")

    # Plot reward metrics
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Performance in 3D Olfactory GridWorld")
    plt.grid(True)
    plt.show()

    # Visualize final trajectory
    plot_scent_heatmap(env)
    plot_3d_trajectory(env)
    # animate_3d_trajectory(env)
    animate_scent_plume(env, wind_vector=(5, 0.5, 0.2), steps=40)


