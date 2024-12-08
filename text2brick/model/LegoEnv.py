import gym
from gym import spaces
import numpy as np

class LegoEnv(gym.Env):

    def __init__(self, size):

        self.size = size

        # Observation space: binary grid of shape (size, size)
        self.observation_space = spaces.MultiBinary([size, size])

        # Action space: coordinates within the grid
        self.action_space = spaces.MultiDiscrete([size, size - 1])
        
        # Initialize the grid with zeros
        self.current_state = np.zeros((size, size), dtype=int)

    def generate_random_action(self):
        return tuple(self.action_space.sample())

    def reset(self):
        """
        Reset the environment to the initial state (all zeros).
        Returns:
            numpy.ndarray: The initial state of the environment.
        """
        self.current_state = np.zeros((self.size, self.size), dtype=int)
        return self.current_state

    def step(self, action):
        """
        Place a 1 in the grid at the given (x, y) coordinate.
        Args:
            action (tuple): A pair of integers (row, col) specifying the cell to update.
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Validate the action using the action space
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Extract row and column from the action
        row, col = action

        # Update the specified cells in the grid to 1
        self.current_state[row, col:col + 2] = 1

        # Placeholder reward and done flag
        reward = 0
        done = False
        info = {}

        return self.current_state, reward, done, info
