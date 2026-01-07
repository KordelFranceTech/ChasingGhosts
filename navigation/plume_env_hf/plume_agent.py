"""
Expected SARSA agent with a neural-network Q-function (PyTorch).

Usage:
- Ensure you have torch and your GaussianPlumeEnv available in scope.
- Instantiate env = GaussianPlumeEnv(cfg)
- Instantiate agent = ExpectedSarsaAgent(env, ...)
- Train with agent.train(num_episodes=..., ...)
"""

import math
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# If your env lives in another file, import it:
from plume_generator import GaussianPlumeEnv, PlumeEnvConfig

# -------------------------------
# Small convolutional Q-network
# -------------------------------
class ConvQNetwork(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        # a light CNN that reduces spatial dims reasonably for moderate grid sizes
        # adapt kernels/strides if your grids are very small/very large
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2)  # -> /2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)           # -> /4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)          # -> /8
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(128 * 4 * 4, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc(x))
        return self.head(x)  # raw Q-values


# -------------------------------
# Expected SARSA Agent
# -------------------------------
class ExpectedSarsaAgent:
    def __init__(
        self,
        env,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.05,
        epsilon_decay_steps: int = 20000,
        device: str = None,
    ):
        """
        env: an instance of GaussianPlumeEnv (or any gym.Env with discrete actions and obs shape (C,H,W))
        """
        self.env = env
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = gamma

        # action space
        self.n_actions = env.action_space.n

        # observation shape
        obs_shape = env.observation_space.shape  # (C, H, W)
        assert len(obs_shape) == 3, "Observation must be (C, H, W)"
        in_channels = obs_shape[0]

        # network + optimizer
        self.qnet = ConvQNetwork(in_channels, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)

        # epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay_steps = epsilon_decay_steps
        self._epsilon_step = 0

        # bookkeeping
        self.loss_fn = nn.MSELoss(reduction="none")  # we'll mask to only update taken action

    def _update_epsilon(self):
        if self._epsilon_step < self.epsilon_decay_steps:
            frac = self._epsilon_step / max(1, self.epsilon_decay_steps)
            self.epsilon = self.epsilon_start + frac * (self.epsilon_final - self.epsilon_start)
        else:
            self.epsilon = self.epsilon_final
        self._epsilon_step += 1

    def select_action(self, obs: np.ndarray) -> int:
        """
        Epsilon-greedy action selection. obs is numpy (C,H,W).
        """
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)  # (1,C,H,W)
                q = self.qnet(t)  # (1, n_actions)
                return int(torch.argmax(q, dim=1).item())

    def _expected_q_next(self, q_next: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        q_next: shape (batch, n_actions) or (n_actions,) for single sample.
        returns expected Q under epsilon-greedy policy.
        For epsilon-greedy: p(greedy) = 1 - epsilon + epsilon / n_actions,
        p(other) = epsilon / n_actions
        Expected Q = sum_a p(a) * Q(a)
        """
        n = q_next.shape[-1]
        # determine greedy action(s)
        greedy_idx = torch.argmax(q_next, dim=-1, keepdim=True)  # (batch,1)
        # probabilities
        prob_random = epsilon / n
        # create a probs tensor equal to epsilon/n for all actions
        probs = torch.full_like(q_next, prob_random)
        # add (1 - epsilon) to greedy action
        if q_next.dim() == 1:
            probs[greedy_idx.item()] += (1.0 - epsilon)
        else:
            # batch case
            batch_idx = torch.arange(q_next.shape[0], device=q_next.device)
            probs[batch_idx, greedy_idx.squeeze(1)] += (1.0 - epsilon)
        expected = (probs * q_next).sum(dim=-1)
        return expected  # shape (batch,) or scalar

    def train_step(self, obs, action: int, reward: float, next_obs, done: bool):
        """
        Run one Expected-SARSA gradient update using a single transition.
        obs, next_obs: numpy arrays (C,H,W)
        action: int
        """
        self._update_epsilon()

        # to tensors
        s = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)       # (1,C,H,W)
        s_next = torch.from_numpy(next_obs).float().unsqueeze(0).to(self.device)
        a = torch.tensor([action], dtype=torch.long, device=self.device)    # (1,)

        q_s = self.qnet(s)             # (1, n_actions)
        q_sa = q_s.gather(1, a.unsqueeze(1)).squeeze(1)  # (1,)

        with torch.no_grad():
            q_next = self.qnet(s_next)    # (1, n_actions)
            if done:
                target = torch.tensor([reward], dtype=torch.float32, device=self.device)
            else:
                # expected value under current epsilon-greedy policy (use current epsilon)
                exp_q_next = self._expected_q_next(q_next.squeeze(0), self.epsilon)
                target = torch.tensor([reward], dtype=torch.float32, device=self.device) + self.gamma * exp_q_next

        # loss only for the taken action
        loss = F.mse_loss(q_sa, target)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping optional
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def evaluate_policy(self, env, episodes: int = 5, render: bool = False) -> float:
        """
        Runs the current (greedy) policy for a few episodes and returns average steps to success or average return.
        """
        total_reward = 0.0
        success_count = 0
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            ep_ret = 0.0
            steps = 0
            while not done:
                # greedy selection (epsilon=0)
                with torch.no_grad():
                    t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                    q = self.qnet(t)
                    action = int(torch.argmax(q, dim=1).item())
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                obs = next_obs
                ep_ret += reward
                steps += 1
                if render:
                    env.render()
                if steps > env.cfg.max_episode_steps:
                    break
            total_reward += ep_ret
            if info.get("reason") == "reached_source":
                success_count += 1
        avg_return = total_reward / episodes
        success_rate = success_count / episodes
        return avg_return, success_rate

    def train(self, num_episodes: int = 500, max_steps_per_episode: int = None, report_every: int = 10):
        """
        Main training loop. On-policy single-step Expected SARSA updates.
        """
        env = self.env
        max_steps = max_steps_per_episode or env.cfg.max_episode_steps

        for ep in range(1, num_episodes + 1):
            obs, _ = env.reset()
            done = False
            ep_loss = 0.0
            ep_reward = 0.0
            steps = 0

            while not done and steps < max_steps:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                loss = self.train_step(obs, action, reward, next_obs, done)
                obs = next_obs
                ep_loss += loss
                ep_reward += reward
                steps += 1

            if ep % report_every == 0 or ep == 1:
                avg_ret, success = self.evaluate_policy(env, episodes=3, render=False)
                print(f"Episode {ep:4d} | steps={steps:3d} ep_ret={ep_reward:6.2f} avg_loss={ep_loss/max(1,steps):.4f} eval_ret={avg_ret:.2f} success={success:.2%} epsilon={self.epsilon:.3f}")

# -------------------------------
# Example: run training on your plume env
# -------------------------------
if __name__ == "__main__":
    import time
    # create a moderate-sized env for faster training during debugging
    try:
        # if you defined PlumeEnvConfig and GaussianPlumeEnv in same namespace
        cfg = None
        # from gaussian_plume_env import PlumeEnvConfig, GaussianPlumeEnv  # try local import
        cfg = PlumeEnvConfig(
            grid_size=(64, 64),
            create_3d=False,        # start with 2D for speed; change to True for 3D
            diffusion=1.0,
            sparsity=0.2,
            n_obstacles=2,
            normalize=True,
            random_seed=0,
            max_episode_steps=300,
        )
        env = GaussianPlumeEnv(cfg)
    except Exception as e:
        print("Could not import GaussianPlumeEnv automatically. Make sure it's in scope.", e)
        raise

    agent = ExpectedSarsaAgent(env, lr=3e-4, gamma=0.98, epsilon_start=1.0, epsilon_final=0.05, epsilon_decay_steps=10000)
    start = time.time()
    agent.train(num_episodes=200, report_every=20)
    print("Training finished in {:.1f}s".format(time.time() - start))

    # quick evaluation
    avg_ret, success_rate = agent.evaluate_policy(env, episodes=10, render=False)
    print(f"Final evaluation: avg_return={avg_ret:.2f}, success_rate={success_rate:.2%}")
