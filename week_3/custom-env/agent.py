import gymnasium as gym
import numpy as np
import random
from collections import defaultdict


class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_decay=0.99):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.learning_rate = alpha
        self.discount_factor = gamma
        self.exploration_rate = epsilon
        self.exploration_decay = epsilon_decay

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * \
            self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]

        self.q_table[state][action] += self.learning_rate * td_error

    def train(self, episodes):
        for episode in range(episodes):
            state, _ = self.env.reset()
            state = (tuple(state["agent"]), tuple(state["target"]))
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)
                next_state = (tuple(next_state["agent"]), tuple(
                    next_state["target"]))

                self.learn(state, action, reward, next_state)
                state = next_state
                done = terminated or truncated

            self.exploration_rate *= self.exploration_decay
        print("training complete")

    def evaluate(self, episodes):
        total_rewards = []
        successes = 0
        total_steps = 0

        original_exploration_rate = self.exploration_rate
        self.exploration_rate = 0  # Disable exploration

        for episode in range(episodes):
            state, _ = self.env.reset()
            state = (tuple(state["agent"]), tuple(state["target"]))
            done = False
            episode_reward = 0
            steps = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)
                next_state = (tuple(next_state["agent"]), tuple(
                    next_state["target"]))

                episode_reward += reward
                state = next_state
                done = terminated or truncated
                steps += 1

                if terminated:
                    successes += 1

            total_rewards.append(episode_reward)
            total_steps += steps

        self.exploration_rate = original_exploration_rate  # Restore exploration rate

        average_reward = np.mean(total_rewards)
        success_rate = successes / episodes
        average_steps = total_steps / episodes

        return average_reward, success_rate, average_steps


class PotentialBasedAgent:
    def __init__(self, env, epsilon=0.1):
        self.env = env
        self.epsilon = epsilon

    def potential(self, state):
        agent_pos, target_pos = state["agent"], state["target"]
        potential_value = -np.sum(np.abs(agent_pos - target_pos))

        # Adjust potential value based on the presence of holes
        for hole_pos in self.env.unwrapped.holes:
            if np.array_equal(agent_pos, hole_pos):
                potential_value -= 10  # Adjust the penalty as needed

        return potential_value

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Choose random action
        else:
            potential_actions = []
            for action in range(self.env.action_space.n):
                direction = self.env.unwrapped.action_to_direction[action]
                new_agent_location = np.clip(
                    state["agent"] + direction, 0, self.env.unwrapped.size - 1)
                new_state = {"agent": new_agent_location,
                             "target": state["target"]}
                potential_actions.append((action, self.potential(new_state)))
            best_action = max(potential_actions, key=lambda x: x[1])[0]
            return best_action

    def evaluate(self, episodes):
        total_rewards = []
        successes = 0
        total_steps = 0

        for episode in range(episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)
                state = next_state
                episode_reward += reward
                done = terminated or truncated
                steps += 1

                if terminated and reward == 10:
                    successes += 1

            total_rewards.append(episode_reward)
            total_steps += steps

        average_reward = np.mean(total_rewards)
        success_rate = successes / episodes
        average_steps = total_steps / episodes

        return average_reward, success_rate, average_steps
