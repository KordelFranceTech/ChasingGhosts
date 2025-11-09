import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

# -------------------------------------------------
# Environment (unchanged)
# -------------------------------------------------
class PlumeEnv:
    def __init__(self, grid_size=20, source_pos=(15, 15)):
        self.grid_size = grid_size
        self.source_pos = source_pos
        self.uav_pos = (5, 5)
        self.done = False
        self.max_steps = 200
        self.steps = 0
        self.plume = np.zeros((grid_size, grid_size))
        self._generate_plume()

    def _generate_plume(self):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                dist = np.sqrt((x - self.source_pos[0])**2 + (y - self.source_pos[1])**2)
                if y <= self.source_pos[1]:
                    self.plume[x, y] = max(0, 10 - dist / 2)
                else:
                    self.plume[x, y] = max(0, 5 - dist / 3)

    def reset(self):
        self.uav_pos = (random.randint(0, self.grid_size-1),
                        random.randint(0, self.grid_size-1))
        self.done = False
        self.steps = 0
        return self._get_state()

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True

        dx, dy = 0, 0
        if action == 0:   dy = 1
        elif action == 1: dx, dy = -1, 1
        elif action == 2: dx, dy =  1, 1
        elif action == 3: dx = -2
        elif action == 4: dx =  2
        elif action == 5: dx, dy = -3, 3
        elif action == 6: dx, dy =  3, 3

        self.uav_pos = (
            max(0, min(self.grid_size-1, self.uav_pos[0] + dx)),
            max(0, min(self.grid_size-1, self.uav_pos[1] + dy))
        )
        self.steps += 1
        reward = -0.01

        if np.linalg.norm(np.array(self.uav_pos) - np.array(self.source_pos)) < 1:
            reward = 10
            self.done = True
        elif self.steps >= self.max_steps:
            self.done = True

        prev_conc = self._get_concentration(self.uav_pos[0]-dx, self.uav_pos[1]-dy)
        curr_conc = self._get_concentration(*self.uav_pos)
        if curr_conc > 0: reward += 0.1
        if curr_conc > prev_conc: reward += 0.2

        return self._get_state(), reward, self.done

    def _get_concentration(self, x, y):
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return self.plume[x, y]
        return 0

    def _get_sensor_readings(self):
        left  = self._get_concentration(self.uav_pos[0] - 1, self.uav_pos[1])
        right = self._get_concentration(self.uav_pos[0] + 1, self.uav_pos[1])
        return left, right

    def _get_state(self):
        left, right = self._get_sensor_readings()
        total = left + right
        diff  = left - right

        plume_status = 0
        if total > 0:   plume_status = 1
        if total >= 5:  plume_status = 2

        gradient = 0
        if abs(diff) >= 1:
            gradient = 1 if diff > 0 else 2

        return plume_status * 3 + gradient   # 0-8


# -------------------------------------------------
# Agent – Q(λ) with eligibility traces
# -------------------------------------------------
class QLambdaAgent:
    def __init__(self, num_states=9, num_actions=7,
                 lr=0.1, gamma=0.9, epsilon=1.0,
                 epsilon_decay=0.999, lambda_=0.9):
        self.q_table = np.zeros((num_states, num_actions))
        self.eligibility = np.zeros((num_states, num_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.lambda_ = lambda_
        self.num_actions = num_actions

    def choose_action(self, state, greedy=False):
        """ε-greedy (greedy=True → always pick best)"""
        if greedy or random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        return int(np.argmax(self.q_table[state]))

    def reset_eligibility(self):
        self.eligibility.fill(0)

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            next_greedy_action = np.argmax(self.q_table[next_state])
            target = reward + self.gamma * self.q_table[next_state, next_greedy_action]

        delta = target - self.q_table[state, action]

        self.eligibility *= self.gamma * self.lambda_
        self.eligibility[state, action] = 1.0
        self.q_table += self.lr * delta * self.eligibility


# -------------------------------------------------
# Agent – Expected SARSA(λ) with eligibility traces
# -------------------------------------------------
class ExpectedSarsaLambdaAgent:
    def __init__(self, num_states=9, num_actions=7,
                 lr=0.1, gamma=0.9, epsilon=1.0,
                 epsilon_decay=0.999, lambda_=0.9):
        self.q_table = np.zeros((num_states, num_actions))
        self.eligibility = np.zeros((num_states, num_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.lambda_ = lambda_
        self.num_actions = num_actions

    def choose_action(self, state, greedy=False):
        """ε-greedy (greedy=True → always pick best)"""
        if greedy or random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        return int(np.argmax(self.q_table[state]))

    def reset_eligibility(self):
        self.eligibility.fill(0)

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            q_next = self.q_table[next_state]
            q_max = np.max(q_next)
            q_sum = np.sum(q_next)
            expected = (self.epsilon / self.num_actions) * q_sum + (1 - self.epsilon) * q_max
            target = reward + self.gamma * expected

        delta = target - self.q_table[state, action]

        self.eligibility *= self.gamma * self.lambda_
        self.eligibility[state, action] = 1.0
        self.q_table += self.lr * delta * self.eligibility


# -------------------------------------------------
# Training
# -------------------------------------------------
def train(episodes=2000):
    env = PlumeEnv()
    agent = ExpectedSarsaLambdaAgent()
    # agent = QLambdaAgent()
    reward_list: list = []

    for ep in range(episodes):
        state = env.reset()
        agent.reset_eligibility()
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        agent.decay_epsilon()
        reward_list.append(total_reward)

        if (ep + 1) % 500 == 0:
            print(f"Episode {ep+1:4d} | ε={agent.epsilon:.3f} | reward={total_reward:.3f}")

    plt.plot(reward_list)
    plt.show()

    return agent, env


# -------------------------------------------------
# Greedy test + path collection
# -------------------------------------------------
def test_and_collect_path(agent, env):
    state = env.reset()
    path = [env.uav_pos]                 # list of (x,y) positions
    actions = []
    states  = [state]

    done = False
    while not done:
        action = agent.choose_action(state, greedy=True)
        next_state, _, done = env.step(action)
        path.append(env.uav_pos)
        actions.append(action)
        states.append(next_state)
        state = next_state

    return path, actions, states


# -------------------------------------------------
# Visualisation
# -------------------------------------------------
def visualise(agent, env, path):
    fig = plt.figure(figsize=(14, 6))

    # ---- 1. Plume + trajectory -------------------------------------------------
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(env.plume.T, cmap='viridis', origin='lower',
                    extent=[0, env.grid_size, 0, env.grid_size])
    ax1.set_title("Plume Concentration & UAV Path")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    plt.colorbar(im, ax=ax1, label='Concentration')

    # start & source
    ax1.plot(*env.source_pos, 'r*', markersize=15, label='Source')
    start = path[0]
    ax1.plot(*start, 'go', markersize=10, label='Start')

    # trajectory line
    xs, ys = zip(*path)
    ax1.plot(xs, ys, 'c-', linewidth=2, label='Path')
    ax1.legend()

    # ---- 2. Q-values per state -------------------------------------------------
    ax2 = fig.add_subplot(1, 2, 2)
    actions_names = ["Fwd", "L-45°", "R-45°", "Hard-L", "Hard-R", "Zig-L", "Zig-R"]
    x = np.arange(agent.q_table.shape[0])
    width = 0.12
    for a in range(agent.q_table.shape[1]):
        ax2.bar(x + a*width, agent.q_table[:, a], width, label=actions_names[a])

    ax2.set_xlabel("State (0-8)")
    ax2.set_ylabel("Q-value")
    ax2.set_title("Learned Q-values")
    ax2.set_xticks(x + 3*width)
    ax2.set_xticklabels([f"S{i}" for i in range(9)])
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    print("Training Expected SARSA(λ)...")
    trained_agent, trained_env = train(episodes=10000)

    print("\nRunning greedy test episode...")
    path, _, _ = test_and_collect_path(trained_agent, trained_env)

    print(f"Path length: {len(path)} steps")
    print("Visualising...")
    visualise(trained_agent, trained_env, path)