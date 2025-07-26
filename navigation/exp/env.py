import gym
import numpy as np
from gym import spaces
import random

class Plume3DEnv(gym.Env):
    def __init__(self, grid_size=20, gas_decay=0.95, diffusion_rate=0.1, source=(10,10,10)):
        super(Plume3DEnv, self).__init__()
        self.grid_size = grid_size
        self.gas_decay = gas_decay
        self.diffusion_rate = diffusion_rate
        self.source = np.array(source)
        self.max_steps = 5000

        # Actions: Up, Down, Forward, Backward, Left, Right
        self.action_space = spaces.Discrete(6)

        # Observation: 3D position and current concentration
        self.observation_space = spaces.Box(
            low=0, high=grid_size,
            shape=(4,), dtype=np.float32
        )

        self._create_environment()
        self.reset()

    def _create_environment(self):
        self.grid = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        self.obstacles = np.zeros_like(self.grid, dtype=bool)

        # Add random obstacles
        for _ in range(int(self.grid_size ** 2.5)):
            x, y, z = np.random.randint(0, self.grid_size, size=3)
            if not np.array_equal((x, y, z), self.source):
                self.obstacles[x, y, z] = True

    def _update_gas(self):
        new_grid = self.grid * self.gas_decay
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                for dz in [-1,0,1]:
                    if dx == dy == dz == 0: continue
                    new_grid += np.roll(self.grid, shift=(dx, dy, dz), axis=(0,1,2)) * self.diffusion_rate
        new_grid[self.source[0], self.source[1], self.source[2]] += 1.0
        self.grid = new_grid

    def reset(self):
        self._create_environment()
        self.grid = np.zeros_like(self.grid)
        self.steps = 0
        self._update_gas()
        while True:
            self.agent_pos = np.random.randint(0, self.grid_size, size=3)
            if not self.obstacles[tuple(self.agent_pos)]: break
        return self._get_obs()

    def _get_obs(self):
        concentration = self.grid[tuple(self.agent_pos)]
        return np.append(self.agent_pos / self.grid_size, concentration)

    def step(self, action):
        moves = [np.array([0,0,1]), np.array([0,0,-1]),
                 np.array([0,1,0]), np.array([0,-1,0]),
                 np.array([1,0,0]), np.array([-1,0,0])]
        next_pos = self.agent_pos + moves[action]
        next_pos = np.clip(next_pos, 0, self.grid_size-1)

        if self.obstacles[tuple(next_pos)]:
            reward = -1.0  # Penalty for hitting obstacle
        else:
            self.agent_pos = next_pos
            reward = self.grid[tuple(self.agent_pos)]

        self._update_gas()
        self.steps += 1
        done = np.linalg.norm(self.agent_pos - self.source) < 1.5 or self.steps >= self.max_steps
        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        pass  # Optional: You can integrate 3D rendering here
