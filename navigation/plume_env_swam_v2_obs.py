import gym
import numpy as np
from gym import spaces
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class PlumeEnvironment3D(gym.Env):
    def __init__(self, grid_size=(20, 20, 10), source_position=(10, 10, 5), gas_diffusivity=0.01, wind_vector=(0.1, 0.0, 0.0), num_agents=5, obstacles=None):
        super(PlumeEnvironment3D, self).__init__()
        self.grid_size = grid_size
        self.source_position = source_position
        self.gas_diffusivity = gas_diffusivity
        self.wind_vector = np.array(wind_vector)
        self.num_agents = num_agents
        self.obstacles = obstacles if obstacles is not None else []

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3 + self.num_agents * 3,), dtype=np.float32)

        self.reset()

    def _generate_concentration_field(self):
        x, y, z = np.indices(self.grid_size)
        sx, sy, sz = self.source_position
        dx = x - sx
        dy = y - sy
        dz = z - sz
        distance_squared = dx**2 + dy**2 + dz**2
        sigma = self.gas_diffusivity * 100
        field = np.exp(-distance_squared / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        return field / np.max(field)

    def reset(self):
        self.agent_positions = [[0, 0, 0] for _ in range(self.num_agents)]
        self.paths = [[(0, 0, 0)] for _ in range(self.num_agents)]
        self.concentration_field = self._generate_concentration_field()
        return [self._get_obs(i) for i in range(self.num_agents)]

    def _get_obs(self, agent_idx):
        own_pos = np.array(self.agent_positions[agent_idx], dtype=np.float32)
        normalized = own_pos / np.array(self.grid_size, dtype=np.float32)
        shared = []
        for i, pos in enumerate(self.agent_positions):
            if i != agent_idx:
                shared.extend(np.array(pos, dtype=np.float32) / np.array(self.grid_size, dtype=np.float32))
        return np.concatenate((normalized, shared))

    def _is_obstacle(self, pos):
        return tuple(pos) in self.obstacles

    def step(self, actions):
        total_reward = 0
        dones = []
        observations = []

        for i, action in enumerate(actions):
            x, y, z = self.agent_positions[i]
            new_pos = [x, y, z]
            if action == 0 and z < self.grid_size[2] - 1:
                new_pos[2] += 1
            elif action == 1 and z > 0:
                new_pos[2] -= 1
            elif action == 2 and y < self.grid_size[1] - 1:
                new_pos[1] += 1
            elif action == 3 and y > 0:
                new_pos[1] -= 1
            elif action == 4 and x > 0:
                new_pos[0] -= 1
            elif action == 5 and x < self.grid_size[0] - 1:
                new_pos[0] += 1

            if not self._is_obstacle(new_pos):
                self.agent_positions[i] = new_pos

            self.paths[i].append(tuple(self.agent_positions[i]))
            x, y, z = self.agent_positions[i]
            conc = self.concentration_field[x, y, z]
            total_reward += conc
            done = self.agent_positions[i] == list(self.source_position)
            dones.append(done)
            observations.append(self._get_obs(i))

        cooperative_reward = total_reward / self.num_agents
        rewards = [cooperative_reward for _ in range(self.num_agents)]

        return observations, rewards, dones, {}

    def render(self, mode='human'):
        for i, pos in enumerate(self.agent_positions):
            print(f"Drone {i} position: {pos}, Concentration: {self.concentration_field[tuple(pos)]}")

    def visualize(self):
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.set_title("Plume Concentration")
        x, y, z = np.indices(self.grid_size)
        values = self.concentration_field
        ax1.scatter(x, y, z, c=values.flatten(), alpha=0.3, cmap='plasma')
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        ax2 = fig.add_subplot(132, projection='3d')
        ax2.set_title("Swarm Drone Paths & Obstacles")
        for path in self.paths:
            path = np.array(path)
            ax2.plot(path[:, 0], path[:, 1], path[:, 2], marker='o')
        ax2.scatter(*self.source_position, color='red', s=100, label='Source')
        if self.obstacles:
            obs_x, obs_y, obs_z = zip(*self.obstacles)
            ax2.scatter(obs_x, obs_y, obs_z, color='black', s=50, label='Obstacles')
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.legend()

        ax3 = fig.add_subplot(133)
        final_positions = np.array(self.agent_positions)
        dists = np.linalg.norm(final_positions[:, np.newaxis, :] - final_positions[np.newaxis, :, :], axis=-1)
        sns.heatmap(dists, annot=True, fmt=".1f", cmap='coolwarm', ax=ax3)
        ax3.set_title("Inter-Agent Distances")
        ax3.set_xlabel("Agent")
        ax3.set_ylabel("Agent")

        plt.tight_layout()
        plt.show()

def train_swarm(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    shared_q_table = np.zeros(env.grid_size + (env.action_space.n,))

    for ep in range(episodes):
        obs = env.reset()
        states = [tuple((o[:3] * env.grid_size).astype(int)) for o in obs]

        for step in range(200):
            actions = []
            for state in states:
                if random.uniform(0, 1) < epsilon:
                    action = random.randint(0, env.action_space.n - 1)
                else:
                    action = np.argmax(shared_q_table[state])
                actions.append(action)

            next_obs, rewards, dones, _ = env.step(actions)
            next_states = [tuple((o[:3] * env.grid_size).astype(int)) for o in next_obs]

            for i in range(env.num_agents):
                best_next_action = np.max(shared_q_table[next_states[i]])
                shared_q_table[states[i] + (actions[i],)] = (
                    (1 - alpha) * shared_q_table[states[i] + (actions[i],)] +
                    alpha * (rewards[i] + gamma * best_next_action)
                )
                states[i] = next_states[i]

            if any(dones):
                break

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}: swarm updated")

    return shared_q_table

if __name__ == "__main__":
    obstacles = [(5, 5, 2), (6, 5, 2), (7, 5, 2), (8, 5, 2), (9, 5, 2)]
    env = PlumeEnvironment3D(num_agents=5, obstacles=obstacles)
    q_table = train_swarm(env)
    env.visualize()
    env.close()