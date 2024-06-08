import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class CustomGridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        """
        This initializes our custom grid world environment.

        Args: 
            render_mode (str): the mode used to render the environment
            size (int): the size of the grid world
        """
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location will be a tuple with the coordinates (x, y).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size-1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size-1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        # Map the action space to the direction we will walk in if that action is taken.
        self.action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1])
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.holes = set()

    def _generate_holes(self):
        """
        Generate random holes in the grid.
        """
        num_holes = min((self.size ** 2) // 8, self.size ** 2 - 2)
        self.holes = set()

        while len(self.holes) < num_holes:
            hole = tuple(self.np_random.integers(0, self.size, size=2))
            if hole != tuple(self._agent_location) and hole not in self.holes:
                self.holes.add(hole)

    def _get_obs(self):
        """
        Get agent and target location

        Returns:
            obs (dict): agent and target location
        """
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        """
        Get distance between agent and target using manhattan distance method (ord = 1)

        Returns:
            distance (dict): distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        """
        Reset the environment
        """
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(
            0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location
        self._generate_holes()

        while np.array_equal(self._target_location, self._agent_location) or tuple(self._target_location) in self.holes:
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Take one action in the environment

        Args:
            action (int): the action taken by the agent

        Returns:
            observation (dict): agent and target location
            reward (float): reward
            terminated (bool): if the episode is terminated
            truncated (bool): if the episode is truncated
            info (dict): distance between agent and target
        """
        direction = self.action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1)
        terminated = np.array_equal(
            self._agent_location, self._target_location)

        if tuple(self._agent_location) in self.holes:
            reward = -5
            terminated = True
        else:
            reward = 10 if terminated else -0.05 * np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """
        Render the environment
        """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        # The size of a single grid square in pixels
        pix_square_size = (self.window_size / self.size)

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Now we draw the holes
        for hole in self.holes:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * np.array(hole),
                    (pix_square_size, pix_square_size),
                ),
            )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
