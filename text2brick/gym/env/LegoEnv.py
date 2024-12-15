import gym
from gym import spaces
import numpy as np

from text2brick.managers.world.SingleBrickLegoWorldManager import SingleBrickLegoWorldManager
from text2brick.models import BrickRef, BrickGetterEnum
from text2brick.gym.components.RewardFunction import IoUValidityRewardFunc, AbstractRewardFunc


class LegoEnv(gym.Env):
    """
    Custom Gym environment for Lego brick placement on a grid.
    """

    def __init__(self, size, reward_func: AbstractRewardFunc = IoUValidityRewardFunc()):
        """
        Initialize the Lego environment.
        
        Args:
            size (int): Size of the grid (size x size).
            reward_func (AbstractRewardFunc): Reward function to evaluate actions. Defaults to IoUValidityRewardFunc.
        """
        self.size = size
        self.n_step = 0
        self.reward_func = reward_func
        self.lego_world = None

        # Define the observation space as a binary grid (size x size)
        self.observation_space = spaces.MultiBinary([size, size])

        # Define the action space as grid coordinates
        self.action_space = spaces.MultiDiscrete([size, size - 1])

        self.reset()


    def __str__(self):
        """
        String representation of the environment, including spaces and reward function.
        
        Returns:
            str: Description of the environment.
        """
        return (
            f"LegoEnv(Environment Size: {self.size}x{self.size}, \n"
            f"Observation Space: {self.observation_space}, \n"
            f"Action Space: {self.action_space}, \n"
            f"Reward Function: {self.reward_func})"
        )


    def reset(self, initial_state: np.array = None):
        """
        Reset the environment to the initial state.
        
        Args:
            initial_state (np.array): Optional grid to start with. Defaults to an empty grid.
        
        Returns:
            numpy.ndarray: Initial state of the environment.
        """
        self.n_step = 0
        if not initial_state:
            initial_state = np.zeros((self.size, self.size), dtype=np.uint8)

        # Init lego world
        brick_ref = BrickRef(file_id="3003.dat", name="2x2", color=15, h=1, w=2, d=2)
        self.lego_world = SingleBrickLegoWorldManager(
            table=initial_state.tolist(),
            brick_ref=brick_ref,
            world_dimension=(self.size, self.size, 1)
        )


    def step(self, action, *args, **kwargs):
        """
        Take a step in the environment by placing a brick at the specified location.

        Args:
            action (tuple): Grid coordinates (row, col) to place the brick.
            *args: Additional positional arguments for the reward function.
            **kwargs: Additional keyword arguments for the reward function.

        Returns:
            tuple: (observation, reward, done, info)
        """
        row, col = action
        x = col
        y = self.size - 1 - row

        # Place the brick in the environment
        is_brick_valid = self.lego_world.add_brick_from_coord(x, y, self.lego_world.data.brick_ref)
        lego_world_array = self.get_obs()

        # Compute the reward using the reward function
        reward = self.reward_func(world_img=lego_world_array, validity=is_brick_valid, *args, **kwargs)
        self.n_step += 1
        info = {
            "reward": reward,
            "steps": self.n_step,
            "brick": self.lego_world.get_brick((x, y, 0), BrickGetterEnum.COORDS)
        }
        done = False

        return lego_world_array, reward, done, info


    def generate_random_action(self):
        """
        Generate a random valid action from the action space.
        
        Returns:
            tuple: Random grid coordinates (row, col).
        """
        return tuple(self.action_space.sample())


    def get_obs(self):
        """
        Get the current state of the environment as a binary grid.
        
        Returns:
            numpy.ndarray: Current grid representation.
        """
        return self.lego_world.recreate_table_from_world()


    def set_reward_function(self, reward_func: AbstractRewardFunc):
        """
        Update the reward function for the environment.
        
        Args:
            reward_func (AbstractRewardFunc): New reward function to use.
        """
        self.reward_func = reward_func