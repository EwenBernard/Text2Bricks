import unittest
import importlib

"""
TO RUN BEFORE PUSHING

If a new module of file has been created please add it to modules_to_test.
"""



modules_to_test = [
    # Dataset
    "text2brick.dataset.LegoDatasetGenerator",
    "text2brick.dataset.LegoPretrainDataset",
    "text2brick.dataset.MNISTDataset",
    "text2brick.dataset.PreprocessImage",

    # Gym
    "text2brick.gym.AbstractRewardFunc",
    "text2brick.gym.IoUValidityRewardFunc",
    "text2brick.gym.LegoEnv",
    "text2brick.gym.BrickPlacementGNN",
    "text2brick.gym.MLP",
    "text2brick.gym.PositionHead2D",
    "text2brick.gym.SNN",
    "text2brick.gym.CNN",
    "text2brick.gym.Text2Brick_v1",

    # Models
    "text2brick.models.Brick",
    "text2brick.models.BrickRef",
    "text2brick.models.BRICK_UNIT",
    "text2brick.models.GraphLegoWorldData",

    # Utils
    "text2brick.utils.ImageUtils",
    "text2brick.utils.WorldDataUtils",
]


class Test(unittest.TestCase):
    def process(self):
        """
        Run all test methods.
        """
        self._import_test()

    def _import_test(self):
        """
        Dynamically test importing all modules and submodules in 'text2brick'.
        """
        for module in modules_to_test:
            with self.subTest(module=module):
                try:
                    importlib.import_module(module)
                    print(f"Successfully imported {module}")
                except ImportError as e:
                    self.fail(f"Failed to import {module}. Error: {e}")


if __name__ == "__main__":
    unittest.main()
