import random
import gymnasium as gym
import numpy as np

alpha = 0.1
gamma = 0.99
epsilon = 0.1
num_episodes = 10000

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="rgb_array")


def greedy_epsilon(qtable, state):
  if random.uniform(0, 1) < epsilon:
    action = env.action_space.sample()
  else:
    if np.all(qtable[state, :]) == qtable[state, 0]:
      action = env.action_space.sample()
    else:
      action = np.argmax(qtable[state, :])
  return action


def run_env():
  qtable = np.zeros((env.observation_space.n, env.action_space.n))
  rewards = []
  steps = []
  for _ in range(num_episodes):
    state, _ = env.reset()
    done = False
    episode_rewards = 0
    episode_steps = 0

    while not done:
      action = greedy_epsilon(qtable, state)
      new_state, reward, terminated, truncated, info = env.step(action)
      next_best_action = np.argmax(qtable[new_state, :])
      td_target = reward + gamma * qtable[new_state, next_best_action]
      td_error = td_target - qtable[state, action]
      qtable[state, action] += alpha * td_error
      state = new_state
      episode_rewards += reward
      episode_steps += 1
      done = terminated or truncated

    rewards.append(episode_rewards)
    steps.append(episode_steps)

  print("Episodes complete")
  print(f"Mean reward: {np.mean(rewards)}")
  print(f"Max steps: {np.max(steps)}")
  return qtable


def evaluation(qtable, num_trials):
  num_successes = 0
  rewards = []
  steps = []
  for _ in range(num_trials):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0
    while not done:
      action = np.argmax(qtable[state, :])
      new_state, reward, terminated, truncated, info = env.step(action)
      state = new_state
      episode_reward += reward
      episode_steps += 1
      done = terminated or truncated

    rewards.append(episode_reward)
    steps.append(episode_steps)
    if episode_reward == 1:
      num_successes += 1

  print("Evaluation complete")
  print(f"Mean reward: {np.mean(rewards)}")
  print(f"Max steps: {np.max(steps)}")
  print(f"Success rate: {num_successes / num_trials}")


env.close()

qtable = run_env()

evaluation(qtable, 10)
