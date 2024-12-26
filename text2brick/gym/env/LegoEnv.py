import gym
from gym import spaces
import numpy as np
import torch
from typing import Tuple
import subprocess

from text2brick.models import GraphLegoWorldData
from text2brick.gym.components.RewardFunction import IoUValidityRewardFunc, AbstractRewardFunc
from text2brick.dataset.Preprocessing import PreprocessImage


class LegoEnv(gym.Env):
    """
    Custom Gym environment for Lego brick placement on a grid.
    """

    def __init__(self, dim: Tuple[int, int], ldr_filename: str = "test_ldr", reward_func: AbstractRewardFunc = IoUValidityRewardFunc()):
        """
        Initialize the Lego environment.
        
        Args:
            size (int): Size of the grid (size x size).
            reward_func (AbstractRewardFunc): Reward function to evaluate actions. Defaults to IoUValidityRewardFunc.
        """
        self.dim = dim
        self.n_step = 0
        self.reward_func = reward_func
        self.lego_world = None

        self.ldr_filename = ldr_filename
        self.is_rendering_3D = False
        self.ldview_process = None


        # Define the observation space as a binary grid (size x size)
        self.observation_space = spaces.MultiBinary([self.dim[0], self.dim[1]])

        # Define the action space as grid coordinates
        self.action_space = spaces.MultiDiscrete([self.dim[1] - 1, self.dim[0] - 1])

        self.reset()


    def __str__(self):
        return (
            f"LegoEnv(Environment Size: {self.dim[0]}x{self.dim[1]}, \n"
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
            initial_state = np.zeros((self.dim[0], self.dim[1]), dtype=np.uint8)

        self.lego_world = GraphLegoWorldData(initial_state)
        self.is_new = True

        # Erase content or create ldr file
        with open(self.ldr_filename + ".ldr", "w") as file:
            pass
        
        self.close()


    def step(self, action, max_step=10, *args, **kwargs) -> Tuple[torch.tensor, torch.tensor, bool, dict]:
        """
        Take a step in the environment by placing a brick at the specified location.

        Args:
            action (tuple): Grid coordinates (row, col) to place the brick.
            *args: Additional positional arguments for the reward function.
            **kwargs: Additional keyword arguments for the reward function.

        Returns:
            tuple: (observation, reward, done, info)
        """
        col, row = action
        done = False

        # Place the brick in the environment
        is_brick_valid = self.lego_world.add_brick(col, row)
        
        if is_brick_valid:
            self.n_step += 1

        lego_world_array = self.get_obs()

        # Compute the reward using the reward function
        reward = self.reward_func(world_img=lego_world_array, validity=is_brick_valid, *args, **kwargs)

        if self.n_step == max_step:
            done = True

        info = {
            "validity": is_brick_valid,
            "steps": self.n_step,
            "brick": f"{col}, {row}"
        }
        
        lego_world_tensor = self.obs_as_tensor(lego_world_array)
        reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(1)

        return lego_world_tensor, reward, done, info
    
    
    def render(self, print_obs=False):
        # Save the LDraw file
        self.lego_world.save_as_ldraw(self.ldr_filename)

        # Print the observation if required
        if print_obs:
            print(self.get_obs())


    def close(self):
        if self.is_rendering_3D:
            try:
                self.ldview_process.terminate()  # Sends SIGTERM (soft kill)
                self.ldview_process.wait()       # Waits for the process to exit
                print("LDView terminated.")
            except KeyboardInterrupt:
                # If the user presses Ctrl+C, kill the process
                self.ldview_process.kill()       # Sends SIGKILL (hard kill)
                self.ldview_process.wait()       # Ensures the process is properly cleaned up
            
            self.is_rendering_3D = False


    def start_3D_rendering(self, ldview_path):
        if not self.is_rendering_3D:
            command = [ldview_path, f"-Polling=4", self.ldr_filename + ".ldr"]
            print("Start 3D")
            try:
                # Use Popen to start the subprocess without blocking
                self.ldview_process = subprocess.Popen(command)
            except Exception as e:
                print(f"Can't open LDView: {e}")
            else:
                self.is_rendering_3D = True


    def generate_random_action(self) -> Tuple[int, int]:
        return tuple(self.action_space.sample())


    def get_obs(self) -> np.array:
        return self.lego_world.graph_to_table()
    

    def obs_as_tensor(self, obs: np.array=None) -> torch.tensor:
        if obs is not None:
            arr = obs
        else:
            arr = self.get_obs()

        preprocess = PreprocessImage()
        return preprocess(arr).unsqueeze(0)


    def set_reward_function(self, reward_func: AbstractRewardFunc):
        self.reward_func = reward_func