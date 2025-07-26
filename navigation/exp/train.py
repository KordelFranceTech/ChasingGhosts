import matplotlib.pyplot as plt
from env import Plume3DEnv
from dqn_agent import DQNAgent
import numpy as np

env = Plume3DEnv()
agent = DQNAgent(obs_size=4, n_actions=6)
episodes = 50
rewards, losses = [], []
final_path = []

for ep in range(episodes):
    print("Episode: ",ep)
    s = env.reset()
    done = False
    total_reward, ep_loss = 0, []
    path = []

    while not done:
        a = agent.act(s, epsilon=max(0.1, 1 - ep / 200))
        s2, r, done, _ = env.step(a)
        agent.remember(s, a, r, s2, done)
        loss = agent.update()
        if loss: ep_loss.append(loss)
        s = s2
        path.append(env.agent_pos.copy())
        total_reward += r

    rewards.append(total_reward)
    losses.append(np.mean(ep_loss) if ep_loss else 0)
    if ep == episodes - 1:
        final_path = path

# Plots
plt.figure()
plt.plot(rewards, label='Rewards')
plt.plot(losses, label='Loss')
plt.legend()
plt.title('DQN Training')
plt.xlabel('Episode')
plt.ylabel('Value')
plt.grid()
plt.show()

# Final path visualization (projected 3D to 2D)
final_path = np.array(final_path)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(final_path[:, 0], final_path[:, 1], final_path[:, 2], label="Agent Path")
ax.scatter(*env.source, color='red', label='Source')
ax.set_title("Agent Final Trajectory")
ax.legend()
plt.show()
