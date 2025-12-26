"""
Features:
- Configurable 2D or 3D plume generation (3D represented as channels x H x W)
- Parameters: temperature (C), relative humidity (0-1), air_density (kg/m^3)
- Diffusion factor (controls sigma_y/sigma_z growth)
- Sparsity (0: continuous, 1: extremely sparse) implemented as patchy dropout
- Obstacles: place several rectangular / cuboid obstacles
- gymnasium.Env subclass with discrete actions:
    - 2D: 0=up,1=down,2=left,3=right
    - 3D: 0=up,1=down,2=left,3=right,4=forward,5=backward
- Observation: numpy array shaped (C, H, W) for both 2D (C=1) and 3D (C=depth)
- Simple reward: -step_penalty, +goal_reward when agent reaches near source
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass
class PlumeEnvConfig:
    grid_size: Tuple[int, int] = (128, 128)   # (H, W)
    depth_slices: int = 8                     # channels for 3D plume
    create_3d: bool = False                   # False => 2D plume (C=1)
    diffusion: float = 1.0                    # multiplier for sigma coefficients
    sparsity: float = 0.0                     # 0 continuous, up to 0.95 (very patchy)
    temperature_c: float = 20.0               # Celsius
    relative_humidity: float = 0.5            # 0..1
    air_density: float = 1.225                # kg/m^3 (sea level ~1.225)
    wind_speed: float = 1.0                   # m/s - mean wind speed along x-direction
    emission_rate: float = 1.0                # Q in arbitrary units
    source_height: float = 1.0                # meters
    world_extent_m: Tuple[float, float, float] = (100.0, 100.0, 10.0)  # physical size (x,y,z)
    n_obstacles: int = 0                      # number of large obstacles (rectangles / cuboids)
    obstacle_size_ratio: Tuple[float, float, float] = (0.2, 0.2, 0.5)  # fraction of world covered by obstacle dims
    max_episode_steps: int = 1000
    step_penalty: float = -0.01
    goal_reward: float = 10.0
    goal_radius_m: float = 2.0                # near-source threshold in meters
    random_seed: Optional[int] = None
    obstacle_block_value: float = 0.0         # value used inside obstacle cells (no plume)
    normalize: bool = True                    # normalize plume to 0..1

class GaussianPlumeGenerator:
    """
    Produces 2D or 3D plume tensors per a simplified Gaussian plume model with tunable parameters.
    """
    def __init__(self, config: PlumeEnvConfig):
        self.cfg = config
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            random.seed(config.random_seed)

        # Derived grid parameters
        self.H, self.W = config.grid_size
        self.depth = config.depth_slices if config.create_3d else 1
        # world extents in meters
        self.world_x, self.world_y, self.world_z = config.world_extent_m

        # Prepare grid coordinates (x along wind direction)
        # x from 0 (far end) to world_x (source at right end) by convention:
        self.x_coords = np.linspace(0, self.world_x, self.W)
        self.y_coords = np.linspace(-self.world_y/2, self.world_y/2, self.H)
        if config.create_3d:
            # slices along z (height) axis
            self.z_coords = np.linspace(0, self.world_z, self.depth)
        else:
            self.z_coords = np.array([config.source_height])

    def _stability_scaling(self, temperature_c: float, rh: float):
        """
        Simple mapping: affects sigma growth.
        - higher temperature -> increased buoyancy -> increases vertical dispersion (sigma_z)
        - higher humidity -> may slightly reduce mixing (simple negative effect on sigma)
        Returns (scale_y, scale_z)
        """
        # baseline scales
        scale_y = 1.0
        scale_z = 1.0

        # temperature effect: linear scaling about 15°C baseline
        temp_base = 15.0
        delta_t = temperature_c - temp_base
        scale_z *= (1.0 + 0.02 * delta_t)   # 2% per °C effect (approx)
        scale_y *= (1.0 + 0.01 * delta_t)   # weaker effect on lateral spread

        # humidity effect: higher RH reduces evaporation and can reduce turbulent mixing slightly
        scale_y *= (1.0 - 0.15 * (rh - 0.5))  # RH 0.5 baseline
        scale_z *= (1.0 - 0.10 * (rh - 0.5))

        # clamp
        scale_y = max(0.1, scale_y)
        scale_z = max(0.1, scale_z)
        return scale_y, scale_z

    def _sigma_yz(self, x: np.ndarray, diffusion: float, scale_y: float, scale_z: float):
        """
        Very simple parametrization for sigma_y and sigma_z as increasing functions of downwind distance x.
        sigma = a + b * x^c scaled by diffusion and stability scales.
        We'll use constants that yield reasonable shapes.
        """
        # base coefficients (meters)
        a_y, b_y, c_y = 0.5, 0.08 * diffusion, 0.9
        a_z, b_z, c_z = 0.3, 0.05 * diffusion, 1.0

        sigma_y = (a_y + b_y * (x ** c_y)) * scale_y + 1e-6
        sigma_z = (a_z + b_z * (x ** c_z)) * scale_z + 1e-6
        return sigma_y, sigma_z

    def generate(self) -> np.ndarray:
        """
        Returns plume tensor shaped (C, H, W)
        Values are concentration-like, non-negative. If normalize is True, scaled to 0..1.
        """
        cfg = self.cfg
        C = self.depth
        H, W = self.H, self.W
        plume = np.zeros((C, H, W), dtype=np.float32)

        # precompute mesh
        X, Y = np.meshgrid(self.x_coords, self.y_coords)  # shape H x W
        # For each z slice, compute vertical contribution
        for ci, z in enumerate(self.z_coords):
            # compute distances along downwind (x). Ensure x > 0 to avoid singularities.
            x_pos = X.copy()
            x_pos[x_pos <= 0.001] = 0.001

            # stability scaling
            scale_y, scale_z = self._stability_scaling(cfg.temperature_c, cfg.relative_humidity)

            # sigma as function of x
            sigma_y, sigma_z = self._sigma_yz(x_pos, cfg.diffusion, scale_y, scale_z)  # arrays H x W

            # Gaussian plume formula (simplified, steady-state, point source)
            # C(x,y,z) = (Q / (2*pi*sigma_y*sigma_z*U)) * exp(-y^2/(2*sigma_y^2)) * vertical_term
            # vertical_term: two image sources to account for ground reflection:
            Hs = cfg.source_height
            vertical_term = np.exp(-((z - Hs) ** 2) / (2 * (sigma_z ** 2))) + np.exp(-((z + Hs) ** 2) / (2 * (sigma_z ** 2)))

            const = cfg.emission_rate / (2.0 * math.pi * sigma_y * sigma_z * max(cfg.wind_speed, 1e-3))
            # elementwise multiplication
            exp_y = np.exp(-(Y ** 2) / (2.0 * (sigma_y ** 2)))
            conc = const * exp_y * vertical_term

            # Clamp and fill
            conc = np.maximum(conc, 0.0)
            plume[ci] = conc

        # apply sparsity: patchy plume. We use a random field low-frequency mask (Perlin-like via gaussian blurred noise)
        if cfg.sparsity > 0.0:
            # create low-frequency noise by upsampling small noise and gaussian blurring (approx)
            small = np.random.rand(max(4, H//16), max(4, W//16)).astype(np.float32)
            # upsample nearest
            up = np.repeat(np.repeat(small, H // small.shape[0] + 1, axis=0)[:H, :], W // small.shape[1] + 1, axis=1)[:, :W]
            # threshold by sparsity to create holes
            mask = (up > cfg.sparsity).astype(np.float32)  # 1 where plume remains
            # broadcast to channels
            plume *= mask[np.newaxis, :, :]

        # obstacles already handled by environment; generator just returns concentrations
        if cfg.normalize:
            max_val = plume.max()
            if max_val > 0:
                plume = plume / max_val

        # optionally clip to [0,1]
        plume = np.clip(plume, 0.0, 1.0)
        return plume


class GaussianPlumeEnv(gym.Env):
    """
    gymnasium.Environment wrapping the plume generator with a navigable agent.
    Observation: ndarray (C, H, W) float32 with values 0..1, with obstacle mask applied.
    Info includes 'agent_pos_m' physical coordinates and 'source_pos_m'.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, config: Optional[PlumeEnvConfig] = None):
        super().__init__()
        self.cfg = config if config is not None else PlumeEnvConfig()
        self.seed_val = self.cfg.random_seed
        if self.seed_val is not None:
            np.random.seed(self.seed_val)
            random.seed(self.seed_val)

        # plume generator
        self.generator = GaussianPlumeGenerator(self.cfg)

        # observation shape (C, H, W)
        self.C = self.generator.depth
        self.H = self.generator.H
        self.W = self.generator.W

        # action space
        if self.cfg.create_3d:
            self.action_map = {
                0: np.array([0, -1, 0]),   # up (y-)
                1: np.array([0, 1, 0]),    # down (y+)
                2: np.array([0, 0, -1]),   # left (x-)
                3: np.array([0, 0, 1]),    # right (x+)
                4: np.array([1, 0, 0]),    # forward (z+)
                5: np.array([-1, 0, 0]),   # backward (z-)
            }
            self.action_space = spaces.Discrete(6)
        else:
            # 2D grid as (y,x)
            self.action_map = {
                0: np.array([-1, 0]),  # up
                1: np.array([1, 0]),   # down
                2: np.array([0, -1]),  # left
                3: np.array([0, 1]),   # right
            }
            self.action_space = spaces.Discrete(4)

        # observation space: floats 0..1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.C, self.H, self.W), dtype=np.float32)

        # agent state
        self._agent_idx = None  # grid index tuple
        self._source_idx = None
        self._plume = None
        self._obs = None
        self._step_count = 0
        self._obstacle_mask = np.ones((self.H, self.W), dtype=np.float32)  # 2D mask; obstacles zero out plume and agent can not enter
        self._z_obstacle_mask = None  # for 3D, (C, H, W) mask
        self._init_obstacles()
        self.reset()

    def _init_obstacles(self):
        # build obstacles as rectangle/cuboid regions set to zero in the obstacle masks
        cfg = self.cfg
        H, W = self.H, self.W
        self._obstacle_mask = np.ones((H, W), dtype=np.float32)
        if cfg.create_3d:
            C = self.C
            self._z_obstacle_mask = np.ones((C, H, W), dtype=np.float32)

        for i in range(cfg.n_obstacles):
            # choose random center
            cy = random.randint(0, H-1)
            cx = random.randint(0, W-1)
            # sizes
            oh = max(3, int(cfg.obstacle_size_ratio[0] * H))
            ow = max(3, int(cfg.obstacle_size_ratio[1] * W))
            y0 = max(0, cy - oh // 2); y1 = min(H, cy + oh // 2)
            x0 = max(0, cx - ow // 2); x1 = min(W, cx + ow // 2)
            self._obstacle_mask[y0:y1, x0:x1] = 0.0
            if cfg.create_3d:
                cz = random.randint(0, self.C-1)
                oz = max(1, int(cfg.obstacle_size_ratio[2] * self.C))
                z0 = max(0, cz - oz // 2); z1 = min(self.C, cz + oz // 2)
                self._z_obstacle_mask[z0:z1, y0:y1, x0:x1] = 0.0

    def _grid_to_world(self, idx: Tuple[int, ...]) -> Tuple[float, float, float]:
        """
        Convert grid indices to world coordinates in meters.
        idx: (y, x) for 2D, or (z_idx, y, x) for internal agent storage for 3D (we will use (y,x) or (z,y,x) conventions).
        Returns (x_m, y_m, z_m)
        """
        if self.cfg.create_3d:
            # agent stored as (z_idx, y_idx, x_idx)
            z_idx, y_idx, x_idx = idx
            x_m = self.generator.x_coords[x_idx]
            y_m = self.generator.y_coords[y_idx]
            z_m = self.z_index_to_m(z_idx)
            return x_m, y_m, z_m
        else:
            y_idx, x_idx = idx
            x_m = self.generator.x_coords[x_idx]
            y_m = self.generator.y_coords[y_idx]
            z_m = self.cfg.source_height
            return x_m, y_m, z_m

    def z_index_to_m(self, z_idx: int):
        # map channel index to z coordinate
        return self.generator.z_coords[z_idx]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self._step_count = 0
        self._plume = self.generator.generate()
        # apply obstacle masks
        if self.cfg.create_3d and self._z_obstacle_mask is not None:
            self._plume *= self._z_obstacle_mask
        else:
            self._plume *= self._obstacle_mask[np.newaxis, :, :]

        # Decide source location: we choose source at downwind edge (rightmost column)
        src_x_idx = self.W - 1
        src_y_idx = self.H // 2
        if self.cfg.create_3d:
            src_z_idx = np.argmin(np.abs(self.generator.z_coords - self.cfg.source_height))
            self._source_idx = (src_z_idx, src_y_idx, src_x_idx)
        else:
            self._source_idx = (src_y_idx, src_x_idx)

        # place agent at far upwind (leftmost column), possibly random y
        start_x_idx = 0
        start_y_idx = self.H // 2
        if self.cfg.create_3d:
            start_z_idx = 0  # bottom slice
            self._agent_idx = (start_z_idx, start_y_idx, start_x_idx)
        else:
            self._agent_idx = (start_y_idx, start_x_idx)

        # ensure agent not in obstacle; if in obstacle, nudge outward
        self._ensure_agent_not_in_obstacle()

        obs = self._get_observation()
        return obs, {}

    def _ensure_agent_not_in_obstacle(self):
        # nudge agent until cell is free (very simple)
        max_try = 1000
        tries = 0
        while self._is_in_obstacle(self._agent_idx) and tries < max_try:
            if self.cfg.create_3d:
                z, y, x = self._agent_idx
                y = min(max(0, y + random.randint(-2, 2)), self.H-1)
                x = min(max(0, x + random.randint(-2, 2)), self.W-1)
                z = min(max(0, z + random.randint(-1, 1)), self.C-1)
                self._agent_idx = (z, y, x)
            else:
                y, x = self._agent_idx
                y = min(max(0, y + random.randint(-2, 2)), self.H-1)
                x = min(max(0, x + random.randint(-2, 2)), self.W-1)
                self._agent_idx = (y, x)
            tries += 1

    def _is_in_obstacle(self, idx):
        if self.cfg.create_3d:
            z, y, x = idx
            return self._z_obstacle_mask[z, y, x] == 0.0
        else:
            y, x = idx
            return self._obstacle_mask[y, x] == 0.0

    def _get_observation(self) -> np.ndarray:
        """
        Returns observation that may include agent overlay (optional) or simply the plume field with obstacles applied.
        Here we return the plume with obstacles already applied.
        """
        # return a copy to prevent external mutation
        return self._plume.copy()

    def step(self, action: int):
        self._step_count += 1
        done = False
        info = {}
        reward = self.cfg.step_penalty

        # compute new agent index
        if self.cfg.create_3d:
            dz, dy, dx = self.action_map[action]
            z, y, x = self._agent_idx
            z_new = int(np.clip(z + dz, 0, self.C-1))
            y_new = int(np.clip(y + dy, 0, self.H-1))
            x_new = int(np.clip(x + dx, 0, self.W-1))
            candidate = (z_new, y_new, x_new)
        else:
            dy, dx = self.action_map[action]
            y, x = self._agent_idx
            y_new = int(np.clip(y + dy, 0, self.H-1))
            x_new = int(np.clip(x + dx, 0, self.W-1))
            candidate = (y_new, x_new)

        # only move if candidate not obstacle
        if not self._is_in_obstacle(candidate):
            self._agent_idx = candidate
        else:
            # collision: small extra penalty
            reward += -0.05

        # check goal (reach near source)
        agent_world = self._grid_to_world(self._agent_idx)
        source_world = self._grid_to_world(self._source_idx)
        dist = math.sqrt((agent_world[0]-source_world[0])**2 + (agent_world[1]-source_world[1])**2 + (agent_world[2]-source_world[2])**2)
        if dist <= self.cfg.goal_radius_m:
            reward += self.cfg.goal_reward
            done = True
            info['reason'] = 'reached_source'

        if self._step_count >= self.cfg.max_episode_steps:
            done = True
            info['reason'] = 'max_steps'

        obs = self._get_observation()
        return obs, float(reward), done, False, info  # gymnasium step signature: obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        if plt is None:
            print("Matplotlib not available; cannot render.")
            return
        if self.cfg.create_3d:
            # render maximum-intensity projection across channels
            mip = self._plume.max(axis=0)
            img = mip
        else:
            img = self._plume[0]

        plt.figure(figsize=(6,6))
        plt.imshow(img, origin='lower', extent=[0, self.generator.world_x, -self.generator.world_y/2, self.generator.world_y/2])
        # plot obstacles
        obs_mask = (self._obstacle_mask == 0.0)
        if obs_mask.any():
            ys, xs = np.where(obs_mask)
            plt.scatter(self.generator.x_coords[xs], self.generator.y_coords[ys], s=1, c='k', alpha=0.3)

        # agent and source
        ax = plt.gca()
        if self.cfg.create_3d:
            z, y, x = self._agent_idx
            ay = self.generator.y_coords[y]; axx = self.generator.x_coords[x]
            sz, sy, sx = self._source_idx
            syy = self.generator.y_coords[sy]; sxx = self.generator.x_coords[sx]
        else:
            ay, axx = self._agent_idx
            ay = self.generator.y_coords[ay]; axx = self.generator.x_coords[axx]
            sy, sx = self._source_idx
            syy = self.generator.y_coords[sy]; sxx = self.generator.x_coords[sx]

        plt.scatter([axx], [ay], c='red', s=40, marker='o', label='agent')
        plt.scatter([sxx], [syy], c='green', s=60, marker='*', label='source')
        plt.legend()
        plt.title("Gaussian plume (MIP for 3D). Agent (red) and source (green)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()

    def close(self):
        pass

# -------------------------------
# Usage example
# -------------------------------
if __name__ == "__main__":
    # example config: 3D sparse plume with obstacles
    cfg = PlumeEnvConfig(
        grid_size=(128, 160),
        depth_slices=12,
        create_3d=False,
        diffusion=1.3,
        sparsity=0.35,
        temperature_c=25.0,
        relative_humidity=0.4,
        air_density=1.18,
        wind_speed=2.0,
        emission_rate=1.0,
        source_height=2.0,
        world_extent_m=(200.0, 120.0, 10.0),
        n_obstacles=0,
        obstacle_size_ratio=(0.12, 0.12, 0.5),
        max_episode_steps=500,
        step_penalty=-0.01,
        goal_reward=5.0,
        goal_radius_m=3.0,
        normalize=True,
        random_seed=42
    )

    env = GaussianPlumeEnv(cfg)
    obs, _ = env.reset()
    print("Observation shape:", obs.shape)  # (C, H, W)

    # sample random actions for a few steps and render
    for i in range(20):
        action = env.action_space.sample()
        obs, rew, done, truncated, info = env.step(action)
        print(f"step {i} reward={rew:.3f} done={done}")
        if i % 5 == 0:
            env.render()
        if done:
            break

    env.close()
