import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_size, n_actions, device="cpu"):
        self.model = DQN(obs_size, n_actions).to(device)
        self.target = DQN(obs_size, n_actions).to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.batch_size = 64
        self.device = device
        self.update_freq = 10
        self.steps = 0

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 5)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.model(state).argmax().item()

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def update(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, d = zip(*batch)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        q_vals = self.model(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target(s2).max(1, keepdim=True)[0]
        q_target = r + (1 - d) * self.gamma * q_next

        loss = nn.MSELoss()(q_vals, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.update_freq == 0:
            self.target.load_state_dict(self.model.state_dict())
        self.steps += 1
        return loss.item()
