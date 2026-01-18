import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum

@dataclass
class PlumeConfig:
    """Configuration parameters for the Gaussian plume model"""
    # Source parameters
    emission_rate: float = 100.0  # g/s
    source_height: float = 10.0   # meters

    # Atmospheric parameters
    wind_speed: float = 5.0       # m/s
    wind_direction: float = 0.0   # degrees (0 = east)
    temperature: float = 293.15   # Kelvin
    humidity: float = 0.5         # relative humidity (0-1)
    air_density: float = 1.2      # kg/m³

    # Dispersion parameters
    stability_class: str = 'D'    # A-G (Pasquill stability classes)
    sparsity_factor: float = 0.5  # 0=continuous, 1=sparse
    diffusion_factor: float = 1.0 # Controls diffusion rate

    # Grid parameters
    x_size: int = 100             # downwind distance (grid points)
    y_size: int = 100             # crosswind distance (grid points)
    z_size: int = 20              # vertical (grid points)
    dx: float = 10.0              # x grid spacing (m)
    dy: float = 10.0              # y grid spacing (m)
    dz: float = 5.0               # z grid spacing (m)

    # 2D/3D mode
    dimensions: int = 3           # 2 or 3

    # Additional tuning factors
    surface_roughness: float = 0.1  # z0 (m)
    lapse_rate: float = -0.006     # K/m (negative = stable)

class StabilityClass(Enum):
    A = 1  # Very unstable
    B = 2
    C = 3
    D = 4  # Neutral
    E = 5
    F = 6  # Stable
    G = 7  # Very stable

class GaussianPlumeEnv(gym.Env):
    """
    Gaussian Plume Model Environment
    Subclassed from Gymnasium for reinforcement learning applications

    2D mode: (x, y) grid [grayscale-like]
    3D mode: (x, y, z_channels) tensor [multi-channel 3D]
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 4}

    def __init__(self, config: PlumeConfig, render_mode: Optional[str] = None):
        super().__init__()

        self.config = config
        self.current_step = 0
        self.max_steps = config.x_size

        # Calculate Pasquill stability parameters
        self.stability_params = self._get_stability_params()

        # Observation space
        if config.dimensions == 2:
            # 2D: (x, y) grid
            obs_shape = (config.y_size, config.x_size)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0,
                shape=obs_shape,
                dtype=np.float32
            )
        else:
            # 3D: (y, x, z_channels)
            obs_shape = (config.y_size, config.x_size, config.z_size)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0,
                shape=obs_shape,
                dtype=np.float32
            )

        # *** FIXED ACTION SPACE ***
        # Continuous Box space - NO discrete 'n' attribute needed
        self.action_space = spaces.Box(
            low=np.array([1.0, -180.0, 0.0]),
            high=np.array([20.0, 180.0, 500.0]),
            dtype=np.float32
        )

        self.render_mode = render_mode

    def _get_stability_params(self) -> Tuple[float, float]:
        """Get sigma_y and sigma_z parameters based on Pasquill stability class"""
        stability_map = {
            StabilityClass.A: (0.22 * self.config.x_size * self.config.dx, 0.20),
            StabilityClass.B: (0.16 * self.config.x_size * self.config.dx, 0.12),
            StabilityClass.C: (0.11 * self.config.x_size * self.config.dx, 0.08),
            StabilityClass.D: (0.08 * self.config.x_size * self.config.dx, 0.06),
            StabilityClass.E: (0.06 * self.config.x_size * self.config.dx, 0.03),
            StabilityClass.F: (0.04 * self.config.x_size * self.config.dx, 0.016),
            StabilityClass.G: (0.04 * self.config.x_size * self.config.dx, 0.012)
        }

        stability = StabilityClass[self.config.stability_class.upper()]
        sigma_y_base, pz = stability_map[stability]

        # Adjust for diffusion factor
        sigma_y = sigma_y_base * self.config.diffusion_factor
        sigma_z = pz * self.config.x_size * self.config.dx * self.config.diffusion_factor

        return sigma_y, sigma_z

    def _gaussian_plume_2d(self, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        """Calculate 2D Gaussian plume concentration"""
        x = x_grid * self.config.dx
        y = y_grid * self.config.dy

        sigma_y, _ = self._get_stability_params()

        # Gaussian plume equation (ground level)
        concentration = (self.config.emission_rate / (np.pi * self.config.wind_speed * sigma_y**2)) * \
                       np.exp(-(y**2) / (2 * sigma_y**2)) * \
                       np.exp(-(x**2) / (2 * (sigma_y * 0.5)**2))  # Simplified downwind spread

        # Apply sparsity factor (intermittent emissions)
        sparsity_mask = np.random.random(concentration.shape) > self.config.sparsity_factor
        concentration[sparsity_mask] *= 0.1  # Reduce sparse regions

        # Normalize to [0, 1]
        return concentration / np.max(concentration)

    def _gaussian_plume_3d(self, x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
        """Calculate 3D Gaussian plume concentration"""
        x = x_grid * self.config.dx
        y = y_grid * self.config.dy
        z = z_grid * self.config.dz

        sigma_y, sigma_z = self._get_stability_params()

        # 3D Gaussian plume equation
        plume_core = np.exp(-(y**2) / (2 * sigma_y**2)) * \
                    np.exp(-((z - self.config.source_height)**2) / (2 * sigma_z**2))

        # Ground reflection (image source)
        plume_reflected = np.exp(-(y**2) / (2 * sigma_y**2)) * \
                         np.exp(-((z + self.config.source_height)**2) / (2 * sigma_z**2))

        concentration_3d = (self.config.emission_rate /
                          (np.pi * self.config.wind_speed * sigma_y * sigma_z * 2)) * \
                         (plume_core + plume_reflected)

        # Apply atmospheric effects
        concentration_3d *= self._atmospheric_correction(x, z)

        # Apply sparsity factor
        sparsity_mask = np.random.random(concentration_3d.shape) > self.config.sparsity_factor
        concentration_3d[sparsity_mask] *= 0.1

        # Normalize each z-slice
        concentration_normalized = np.zeros_like(concentration_3d)
        for i in range(concentration_3d.shape[2]):
            slice_max = np.max(concentration_3d[:, :, i])
            if slice_max > 0:
                concentration_normalized[:, :, i] = concentration_3d[:, :, i] / slice_max

        return concentration_normalized

    def _atmospheric_correction(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Apply temperature, humidity, and density corrections"""
        # Temperature lapse rate effect
        temp_correction = np.exp(self.config.lapse_rate * z / self.config.temperature)

        # Humidity effect on diffusion
        humidity_factor = 1 + 0.2 * self.config.humidity

        # Density effect
        density_factor = 1 / self.config.air_density

        # Roughness effect on vertical mixing
        roughness_factor = np.exp(-z / (self.config.surface_roughness * 10))

        return temp_correction * humidity_factor * density_factor * roughness_factor

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Create coordinate grids
        x_grid = np.arange(self.config.x_size)
        y_grid = np.arange(self.config.y_size)
        X, Y = np.meshgrid(x_grid, y_grid)

        if self.config.dimensions == 2:
            self.state = self._gaussian_plume_2d(X, Y)
        else:
            z_grid = np.arange(self.config.z_size)
            self.state = self._gaussian_plume_3d(X, Y, z_grid)

        return self.state, {}

    def step(self, action):
        """Apply action and advance simulation"""
        # Update parameters from action
        self.config.wind_speed = float(action[0])
        self.config.wind_direction = float(action[1])
        self.config.emission_rate = float(action[2])

        self.current_step += 1

        # Regenerate plume
        x_grid = np.arange(self.config.x_size)
        y_grid = np.arange(self.config.y_size)
        X, Y = np.meshgrid(x_grid, y_grid)

        if self.config.dimensions == 2:
            self.state = self._gaussian_plume_2d(X, Y)
        else:
            z_grid = np.arange(self.config.z_size)
            self.state = self._gaussian_plume_3d(X, Y, z_grid)

        # Rotate plume based on wind direction (simplified)
        if self.config.dimensions == 2:
            angle_rad = np.radians(self.config.wind_direction)
            self.state = np.rot90(self.state, k=int(angle_rad / (np.pi/2)) % 4)

        terminated = self.current_step >= self.max_steps
        truncated = False
        reward = -np.mean(self.state)  # Example: minimize exposure

        return self.state, reward, terminated, truncated, {}

    def render(self):
        """Render the current state"""
        if self.config.dimensions == 2:
            plt.figure(figsize=(10, 8))
            plt.imshow(self.state, cmap='hot', origin='lower')
            plt.colorbar(label='Normalized Concentration')
            plt.title(f'2D Gaussian Plume - Step {self.current_step}')
            plt.xlabel('Downwind Distance')
            plt.ylabel('Crosswind Distance')
            plt.show()
        else:
            # Show vertical slices
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            for i in [0, self.config.z_size//4, self.config.z_size//2, -1]:
                im = axes.flat[i].imshow(self.state[:, :, i], cmap='hot', origin='lower')
                axes.flat[i].set_title(f'Z-level {i}')
                plt.colorbar(im, ax=axes.flat[i])
            plt.tight_layout()
            plt.show()

# Example usage and testing
if __name__ == "__main__":
    # 2D Plume
    config_2d = PlumeConfig(dimensions=2, sparsity_factor=0.3, diffusion_factor=6.5)
    env_2d = GaussianPlumeEnv(config_2d)

    obs, _ = env_2d.reset()
    print(f"2D Observation shape: {obs.shape}")
    print(f"2D Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

    # Take a random action
    action = env_2d.action_space.sample()
    obs, reward, terminated, truncated, info = env_2d.step(action)
    env_2d.render()
    env_2d.close()
