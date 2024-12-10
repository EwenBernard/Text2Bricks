import gym
from gym import spaces
import numpy as np

from text2brick.managers.world.SingleBrickLegoWorldManager import SingleBrickLegoWorldManager
from text2brick.models import BrickRef, BrickGetterEnum
from text2brick.utils.ImageUtils import IoU

class LegoEnv(gym.Env):

    def __init__(self, size):

        self.size = size
        self.alpha = 0.5
        self.n_step = 0

        # Observation space: binary grid of shape (size, size)
        self.observation_space = spaces.MultiBinary([size, size])

        # Action space: coordinates within the grid
        self.action_space = spaces.MultiDiscrete([size, size - 1])

        # Init lego world
        brick_ref = BrickRef(file_id="3003.dat", name="2x2", color=15, h=24, w=20, d=20)
        self.lego_world = SingleBrickLegoWorldManager(table=np.zeros((size, size), dtype=np.uint8).tolist(), brick_ref=brick_ref, world_dimension=(size, size, 1))


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


    def step(self, action, model: np.array):
        """
        Place a 1 in the grid at the given (x, y) coordinate.
        Args:
            action (tuple): A pair of integers (row, col) specifying the cell to update.
            model: To compare the observation
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Add brick to world
        row, col = action
        x = col * self.lego_world.data.x_step
        y = (self.size - 1 - row) * self.lego_world.data.y_step
        is_brick_valid = self.lego_world.add_brick_from_coord(x, y, self.lego_world.data.brick_ref)
        
        # Reward
        if is_brick_valid:
            reward = 1
        else:
            reward = -1
        lego_world_array = self.lego_world.recreate_table_from_world()
        iou = IoU(model, lego_world_array)
        reward = reward + self.alpha * iou

        self.n_step += 1

        done = False
        info = {
            "validity" : is_brick_valid,
            "IoU" : iou,
            "reward" : reward,
            "steps" : self.n_step,
            "brick" : self.lego_world.get_brick((x, y, 0), BrickGetterEnum.COORDS)
        }

        return lego_world_array, reward, done, info