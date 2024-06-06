from pathlib import Path  # to work with file paths
from typing import NamedTuple  # for type safety and for better code documentation

import matplotlib.pyplot as plt  # for plotting purposes
import numpy as np  # to work with arrays
import pandas as pd  # to work with DataFrames
import seaborn as sns  # to make heatmaps
from tqdm import tqdm  # to visualise iterables like loops in the form of a progress bar

import gymnasium as gym  # reinforcement learning library
# to make custom sized maps
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

sns.set_theme()  # setting the deffault theme


class Params(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    proba_frozen: float  # Probability that a tile is frozen
    savefig_folder: Path  # Root folder where plots are saved


params = Params(
    total_episodes=1000,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    map_size=5,
    seed=123,
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
    proba_frozen=0.9,
    savefig_folder=Path("plots"),
)

rng = np.random.default_rng(params.seed)

params.savefig_folder.mkdir(parents=True, exist_ok=True)


class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        """
        Initializes a Q-learning agent.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        """
        Updates the Q-table for a given state-action pair based on the reward and the maximum Q-value of the next state.

        Parameters:
            state (int): The current state.
            action (int): The action taken in the current state.
            reward (float): The reward received after taking the action.
            new_state (int): The next state.

        Returns:
            float: The updated Q-value for the state-action pair.
        """
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        """
        Resets the Q-table to zero.
        """
        self.qtable = np.zeros((self.state_size, self.action_size))

    def adjust_reward(self, reward):
        """
        Reward shaping can be done here in order to improve the performance of the agent. But for now we are using the baseline reward structure.
        """
        return reward


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, action_space, state, qtable):
        """
        Choose an action based on the exploration-exploitation tradeoff.

        Parameters:
            action_space (gym.spaces.Space): The action space of the environment.
            state (int): The current state.
            qtable (numpy.ndarray): The Q-table.

        Returns:
            int: The chosen action.

        This function uses the epsilon-greedy algorithm to choose an action based on the exploration-exploitation tradeoff. If the exploration-exploitation tradeoff is less than the epsilon value, a random action is chosen. Otherwise, if all the Q-values for the current state are equal, a random action is chosen. Otherwise, the action with the highest Q-value is chosen.

        """
        explor_exploit_tradeoff = rng.uniform(0, 1)

        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        else:

            if np.all(qtable[state, :]) == qtable[state, 0]:
                action = action_space.sample()
            else:
                action = np.argmax(qtable[state, :])
        return action


def run_env():
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []

    for run in range(params.n_runs):
        learner.reset_qtable()

        for episode in tqdm(
            episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False
        ):
            state = env.reset(seed=params.seed)[0]
            step = 0
            done = False
            total_rewards = 0

            while not done:
                action = explorer.choose_action(
                    action_space=env.action_space, state=state, qtable=learner.qtable
                )

                all_states.append(state)
                all_actions.append(action)

                new_state, reward, terminated, truncated, info = env.step(
                    action)
                done = terminated or truncated
                adjusted_reward = learner.adjust_reward(reward)

                learner.qtable[state, action] = learner.update(
                    state, action, adjusted_reward, new_state
                )

                total_rewards += adjusted_reward
                step += 1

                state = new_state

            rewards[episode, run] = total_rewards
            steps[episode, run] = step
        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions


def postprocess(episodes, params, rewards, steps, map_size):
    """
    Generate the post-processing of the given data.

    Parameters:
        episodes (array-like): The episodes data.
        params (Params): The parameters object.
        rewards (array-like): The rewards data.
        steps (array-like): The steps data.
        map_size (int): The size of the map.

    Returns:
        tuple: A tuple containing two pandas DataFrames.
            - res (DataFrame): The processed rewards and steps data.
            - st (DataFrame): The processed steps data.
    """
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(),
            "Steps": steps.flatten(),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st


def qtable_directions_map(qtable, map_size):
    """
    Generate a map of the maximum Q-values and the corresponding best actions in a Q-table.

    Parameters:
        qtable (numpy.ndarray): The Q-table.
        map_size (int): The size of the map.

    Returns:
        tuple: A tuple containing two numpy arrays.
            - qtable_val_max (numpy.ndarray): The maximum Q-values for each state in the Q-table.
            - qtable_directions (numpy.ndarray): The best actions represented by arrows for each state in the Q-table.
    """
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:

            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


def plot_q_values_map(qtable, env, map_size):
    """
    Plot the Q-values map for a given Q-table and environment.

    Parameters:
        qtable (numpy.ndarray): The Q-table.
        env (gym.Env): The environment.
        map_size (int): The size of the map.

    Returns:
        None
    """
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    img_title = f"frozenlake_q_values_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


def plot_states_actions_distribution(states, actions, map_size):
    """
    Plot the distribution of states and actions for a given map size.

    Parameters:
        states (array-like): The states data.
        actions (array-like): The actions data.
        map_size (int): The size of the map.

    Returns:
        None
    """
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


map_sizes = [4, 7, 9, 11]
res_all = pd.DataFrame()
st_all = pd.DataFrame()

for map_size in map_sizes:
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=params.is_slippery,
        render_mode="rgb_array",
        desc=generate_random_map(
            size=map_size, p=params.proba_frozen, seed=params.seed
        ),
    )

    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=env.observation_space.n)
    env.action_space.seed(
        params.seed
    )
    learner = Qlearning(
        learning_rate=params.learning_rate,
        gamma=params.gamma,
        state_size=params.state_size,
        action_size=params.action_size,
    )
    explorer = EpsilonGreedy(
        epsilon=params.epsilon,
    )

    print(f"Map size: {map_size}x{map_size}")
    rewards, steps, episodes, qtables, all_states, all_actions = run_env()
    print(f"Average Steps {np.mean(steps)} for map {map_size}x{map_size}")
    print(f"Average Rewards {np.mean(rewards)} for map {map_size}x{map_size}")

    res, st = postprocess(episodes, params, rewards, steps, map_size)
    res_all = pd.concat([res_all, res])
    st_all = pd.concat([st_all, st])
    qtable = qtables.mean(axis=0)

    plot_states_actions_distribution(
        states=all_states, actions=all_actions, map_size=map_size
    )
    plot_q_values_map(qtable, env, map_size)

    env.close()


def plot_steps_and_rewards(rewards_df, steps_df):
    """
    Plot the cumulated rewards and averaged steps number over episodes for different map sizes.

    Parameters:
        rewards_df (pandas.DataFrame): The DataFrame containing the cumulated rewards data.
        steps_df (pandas.DataFrame): The DataFrame containing the averaged steps number data.

    Returns:
        None

    This function creates a figure with two subplots. The first subplot shows the cumulated rewards over episodes for different map sizes. The second subplot shows the averaged steps number over episodes for different map sizes. The figures are saved as "frozenlake_steps_and_rewards.png" in the savefig_folder specified in the params object.
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(
        data=rewards_df, x="Episodes", y="cum_rewards", hue="map_size", ax=ax[0]
    )
    ax[0].set(ylabel="Cumulated rewards")

    sns.lineplot(data=steps_df, x="Episodes",
                 y="Steps", hue="map_size", ax=ax[1])
    ax[1].set(ylabel="Averaged steps number")

    for axi in ax:
        axi.legend(title="map size")
    fig.tight_layout()
    img_title = "frozenlake_steps_and_rewards.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


plot_steps_and_rewards(res_all, st_all)
