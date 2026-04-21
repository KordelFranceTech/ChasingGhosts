import numpy as np
import random
import matplotlib.pyplot as plt

WINDOW: int = 100
EPOCHS: int = 10000
LEARNING_RATE: float = 0.0001
GAMMA: float = 0.9
EPSILON: float = 1.0
EPSILON_DECAY: float = 0.999
LAMBDA: float = 0.8
N_STATES: int = 9
N_ACTIONS: int = 7
USE_BLANKS: bool = True

# todo @kordel.france: add in optimal policy and paste here as table

# Simple 2D grid environment for chemical plume tracking
class PlumeEnv:
    def __init__(self, grid_size=20, source_pos=(15, 15), wind_dir='north'):
        self.grid_size = grid_size
        self.source_pos = source_pos
        self.wind_dir = wind_dir  # Assume north wind for simplicity (plume spreads south)
        self.uav_pos = (5, 5)  # Starting position
        self.done = False
        self.max_steps = 200
        self.steps = 0

        # Simple plume model: Gaussian-like concentration
        self.plume = np.zeros((grid_size, grid_size))
        self._generate_plume()

    def _generate_plume(self):
        # Simulate plume as decaying from source
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                dist = np.sqrt((x - self.source_pos[0]) ** 2 + (y - self.source_pos[1]) ** 2)
                # Higher concentration upwind (assuming north is positive y)
                if y < self.source_pos[1]:  # South of source
                    self.plume[x, y] = max(0, 10 - dist / 2)
                else:
                    self.plume[x, y] = max(0, 5 - dist / 3)  # Weaker downwind

        # Generate random blanks in 20% of the map
        if USE_BLANKS:
            for x in range(int(0.2*self.grid_size)):
                x_rand: int = random.randint(0, self.grid_size - 1)
                y_rand: int = random.randint(0, self.grid_size - 1)
                self.plume[x_rand, y_rand] = 0

    def reset(self):
        self.uav_pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
        self.done = False
        self.steps = 0
        return self._get_state()

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True

        # Actions: 0: forward, 1: left, 2: right, 3: hard left, 4: hard right, 5: zigzag left, 6: zigzag right
        dx, dy = 0, 0
        if action == 0:  # forward (surge ahead)
            dy = 1
        elif action == 1:  # slight left
            dx = -1
            dy = 1
        elif action == 2:  # slight right
            dx = 1
            dy = 1
        elif action == 3:  # hard left
            dx = -2
        elif action == 4:  # hard right
            dx = 2
        elif action == 5:  # zigzag left (cast: left then forward)
            dx = -3
            dy = 3
        elif action == 6:  # zigzag right
            dx = 3
            dy = 3

        # Update position (clip to grid)
        self.uav_pos = (
            max(0, min(self.grid_size - 1, self.uav_pos[0] + dx)),
            max(0, min(self.grid_size - 1, self.uav_pos[1] + dy))
        )

        self.steps += 1
        reward = -0.01  # Small step penalty
        if np.linalg.norm(np.array(self.uav_pos) - np.array(self.source_pos)) < 1:
            reward = 10  # Found source
            self.done = True
        elif self.steps >= self.max_steps:
            self.done = True

        # Shaped rewards: +0.1 if on plume, +0.2 if moving toward higher concentration
        prev_conc = self._get_concentration(self.uav_pos[0] - dx, self.uav_pos[1] - dy)
        curr_conc = self._get_concentration(*self.uav_pos)
        if curr_conc > 0:
            reward += 0.2
        if curr_conc > prev_conc:
            reward += 0.5

        return self._get_state(), reward, self.done

    def _get_concentration(self, x, y):
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return self.plume[x, y]
        return 0

    def _get_sensor_readings(self):
        # Simulate two sensors: left (x-1), right (x+1), assuming facing north (positive y)
        left_conc = self._get_concentration(self.uav_pos[0] - 1, self.uav_pos[1])
        right_conc = self._get_concentration(self.uav_pos[0] + 1, self.uav_pos[1])
        return left_conc, right_conc

    def _get_state(self):
        left, right = self._get_sensor_readings()
        total_conc = left + right
        diff = left - right

        # Discretize
        if total_conc == 0:
            plume_status = 0  # off plume
        elif total_conc < 5:
            plume_status = 1  # low concentration
        else:
            plume_status = 2  # high concentration

        if abs(diff) < 1:
            gradient = 0  # balanced
        elif diff > 0:
            gradient = 1  # left > right
        else:
            gradient = 2  # right > left

        # For simplicity, no recent history in this prototype
        state = plume_status * 3 + gradient  # 0-8 states
        return state


# Q(λ) - Learning Agent with eligibility traces (Watkins' Q(λ) with replacing traces)
class QLambdaAgent:
    def __init__(self,
                 num_states=N_STATES,
                 num_actions=N_ACTIONS,
                 lr=LEARNING_RATE,
                 gamma=GAMMA,
                 epsilon=EPSILON,
                 epsilon_decay=EPSILON_DECAY,
                 lambda_=LAMBDA):
        self.q_table = np.zeros((num_states, num_actions))
        self.eligibility = np.zeros((num_states, num_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.lambda_ = lambda_
        self.num_actions = num_actions

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        return np.argmax(self.q_table[state])

    def reset_eligibility(self):
        self.eligibility.fill(0)

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay


# ========================
# Agent: Expected SARSA(λ) with Eligibility Traces
# ========================
class ExpectedSarsaLambdaAgent:
    def __init__(self,
                 num_states=N_STATES,
                 num_actions=N_ACTIONS,
                 lr=LEARNING_RATE,
                 gamma=GAMMA,
                 epsilon=EPSILON,
                 epsilon_decay=EPSILON_DECAY,
                 lambda_=LAMBDA):
        self.q_table = np.zeros((num_states, num_actions))
        self.eligibility = np.zeros((num_states, num_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.lambda_ = lambda_
        self.num_actions = num_actions

    def choose_action(self, state):
        """ε-greedy action selection (unchanged)"""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        return np.argmax(self.q_table[state])

    def reset_eligibility(self):
        self.eligibility.fill(0)

    def decay_epsilon(self):
        # self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        self.epsilon *= self.epsilon_decay

    def update(self, state, action, reward, next_state, done):
        """Expected SARSA(λ) update with accumulating traces"""
        # Compute expected Q-value for next state under ε-greedy policy
        if done:
            target = reward
        else:
            q_next = self.q_table[next_state]
            q_max = np.max(q_next)
            q_sum = np.sum(q_next)
            expected_q = (self.epsilon / self.num_actions) * q_sum + (1 - self.epsilon) * q_max
            target = reward + self.gamma * expected_q

        # TD error
        delta = target - self.q_table[state, action]

        # Update eligibility trace (accumulating trace)
        self.eligibility *= self.gamma * self.lambda_
        self.eligibility[state, action] = 1.0  # Replace trace for current (s,a)

        # Update Q-table
        self.q_table += self.lr * delta * self.eligibility


# Training function (eligibility traces integrated here)
def train_expected_sarsa(episodes=EPOCHS, should_plot=True):
    env = PlumeEnv()
    reward_list: list = []
    agent = ExpectedSarsaLambdaAgent()

    for ep in range(episodes):
        state = env.reset()
        agent.reset_eligibility()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        if ep % 100 == 0:
            print(f"Episode {ep}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
        reward_list.append(total_reward)

    if should_plot:
        plt.plot(reward_list)
        plt.show()

    return agent, env, reward_list


def train_q_lambda(episodes=EPOCHS, should_plot=True):
    env = PlumeEnv()
    agent = QLambdaAgent(num_states=9, num_actions=7)  # 9 states (3 plume x 3 grad), 7 actions
    reward_list: list = []

    for ep in range(episodes):
        state = env.reset()
        agent.reset_eligibility()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            greedy_action = np.argmax(agent.q_table[state])
            if action != greedy_action:
                agent.eligibility.fill(0)
            next_state, reward, done = env.step(action)

            if not done:
                next_greedy_action = np.argmax(agent.q_table[next_state])
                target = reward + agent.gamma * agent.q_table[next_state, next_greedy_action]
            else:
                target = reward

            delta = target - agent.q_table[state, action]

            agent.eligibility *= agent.gamma * agent.lambda_
            agent.eligibility[state, action] = 1

            agent.q_table += agent.lr * delta * agent.eligibility

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        if ep % 100 == 0:
            print(f"Episode {ep}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
        reward_list.append(total_reward)

    if should_plot:
        plt.plot(reward_list)
        plt.show()

    return agent, env, reward_list


def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# Run training
if __name__ == "__main__":
    agent, env, reward_list_q = train_q_lambda()
    agent, env, reward_list_ea = train_expected_sarsa()
    print("Q-Table:")
    print(agent.q_table)
    # To test: Reset env, use agent to navigate (choose actions greedily)
    # plt.plot(reward_list_q)
    # plt.plot(reward_list_ea)

    a_q = np.mean(rolling_window(np.array(reward_list_q), WINDOW), axis=-1)
    # rolling var along last axis
    b_q = np.sqrt(np.var(rolling_window(np.array(reward_list_q), WINDOW), axis=-1))
    Q_MU_LOSSES = a_q
    Q_SIGMA_LOSSES = b_q

    a_ea = np.mean(rolling_window(np.array(reward_list_ea), WINDOW), axis=-1)
    # rolling var along last axis
    b_ea = np.sqrt(np.var(rolling_window(np.array(reward_list_ea), WINDOW), axis=-1))
    EA_MU_LOSSES = a_ea
    EA_SIGMA_LOSSES = b_ea

    x = np.arange(len(reward_list_q))
    plt.plot(x, Q_MU_LOSSES, 'r-', label="Q(λ")
    plt.fill_between(x, Q_MU_LOSSES - Q_SIGMA_LOSSES, Q_MU_LOSSES + Q_SIGMA_LOSSES, color='r', alpha=0.2)
    plt.plot(x, EA_MU_LOSSES, 'b-', label="Expected SARSA(λ)")
    plt.fill_between(x, EA_MU_LOSSES - EA_SIGMA_LOSSES, EA_MU_LOSSES + EA_SIGMA_LOSSES, color='b', alpha=0.2)
    plt.title("Average Rewards Over Simulation Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend(["Q(λ) Mean", "Q(λ) Variance", "Expected SARSA(λ) Mean", "Expected SARSA(λ) Variance"])
    plt.show()

