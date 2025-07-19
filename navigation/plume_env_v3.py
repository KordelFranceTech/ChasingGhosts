import gym
import numpy as np
from gym import spaces
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PlumeEnvironment3D(gym.Env):

    def __init__(self, grid_size=(20, 20, 10), source_position=(10, 10, 5), gas_diffusivity=0.001, wind_vector=(0.1, 0.0, 0.0)):
        super(PlumeEnvironment3D, self).__init__()
        self.grid_size = grid_size
        self.source_position = source_position
        self.gas_diffusivity = gas_diffusivity
        self.wind_vector = np.array(wind_vector)

        self.action_space = spaces.Discrete(6)  # up, down, forward, back, left, right
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        self.reset()

    def _generate_concentration_field(self):
        x, y, z = np.indices(self.grid_size)
        sx, sy, sz = self.source_position

        # Gaussian dispersion model
        dx = x - sx
        dy = y - sy
        dz = z - sz
        distance_squared = dx**2 + dy**2 + dz**2

        sigma = self.gas_diffusivity * 100  # scale for visual effect
        field = np.exp(-distance_squared / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        return field / np.max(field)

    def reset(self):
        self.agent_position = [0, 0, 0]
        self.path = [tuple(self.agent_position)]
        self.concentration_field = self._generate_concentration_field()
        return self._get_obs()

    def _get_obs(self):
        return np.array(self.agent_position, dtype=np.float32) / np.array(self.grid_size, dtype=np.float32)

    def step(self, action):
        x, y, z = self.agent_position
        if action == 0 and z < self.grid_size[2] - 1:
            z += 1  # up
        elif action == 1 and z > 0:
            z -= 1  # down
        elif action == 2 and y < self.grid_size[1] - 1:
            y += 1  # forward
        elif action == 3 and y > 0:
            y -= 1  # back
        elif action == 4 and x > 0:
            x -= 1  # left
        elif action == 5 and x < self.grid_size[0] - 1:
            x += 1  # right

        self.agent_position = [x, y, z]
        self.path.append((x, y, z))
        conc = self.concentration_field[x, y, z]

        done = self.agent_position == list(self.source_position)
        reward = 1.0 if done else conc  # encourage moving toward higher concentration

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        print(f"Drone position: {self.agent_position}, Concentration: {self.concentration_field[tuple(self.agent_position)]}")

    def visualize(self):
        fig = plt.figure(figsize=(12, 6))

        # Plot plume
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title("Plume Concentration")
        x, y, z = np.indices(self.grid_size)
        values = self.concentration_field
        ax1.scatter(x, y, z, c=values.flatten(), alpha=0.3, cmap='plasma')
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        # Plot drone path
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title("Drone Path")
        path = np.array(self.path)
        ax2.plot(path[:, 0], path[:, 1], path[:, 2], marker='o', color='green')
        ax2.scatter(*self.source_position, color='red', s=100, label='Source')
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.legend()

        plt.tight_layout()
        plt.show()

# --- Simple RL Training Loop (Q-Learning, Tabular) ---

def train_agent(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_table = np.zeros(env.grid_size + (env.action_space.n,))

    for ep in range(episodes):
        obs = env.reset()
        state = tuple((obs * env.grid_size).astype(int))

        for step in range(200):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            obs, reward, done, _ = env.step(action)
            new_state = tuple((obs * env.grid_size).astype(int))
            best_next_action = np.max(q_table[new_state])

            q_table[state + (action,)] = (1 - alpha) * q_table[state + (action,)] + alpha * (reward + gamma * best_next_action)
            state = new_state

            if done:
                break

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}: reached source in {step+1} steps")

    return q_table

# Usage example:
if __name__ == "__main__":
    env = PlumeEnvironment3D()
    q_table = train_agent(env)
    env.visualize()
    env.close()
