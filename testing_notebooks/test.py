import unittest
import importlib

"""
Please run this test file before pushing: python -u "test.py"

If new modules/files/classes have been created, please add them to the imports_to_test list
"""

imports_to_test = [
    # (Module path, Class or function to test within the module)
    ("text2brick.dataset", "LegoDatasetGenerator"),
    ("text2brick.dataset", "LegoPretrainDataset"),
    ("text2brick.dataset", "MNISTDataset"),
    ("text2brick.dataset", "PreprocessImage"),
    ("text2brick.gym", "AbstractRewardFunc"),
    ("text2brick.gym", "IoUValidityRewardFunc"),
    ("text2brick.gym", "LegoEnv"),
    ("text2brick.gym", "BrickPlacementGNN"),
    ("text2brick.gym", "MLP"),
    ("text2brick.gym", "PositionHead2D"),
    ("text2brick.gym", "SNN"),
    ("text2brick.gym", "CNN"),
    ("text2brick.gym", "Text2Brick_v1"),
    ("text2brick.models", "Brick"),
    ("text2brick.models", "BrickRef"),
    ("text2brick.models", "BRICK_UNIT"),
    ("text2brick.models", "GraphLegoWorldData"),
    ("text2brick.utils.ImageUtils", "array_to_image"),
    ("text2brick.utils.ImageUtils", "image_upscale"),
    ("text2brick.utils.ImageUtils", "IoU"),
    ("text2brick.utils.WorldDataUtils", "save_ldr"),
    ("text2brick.utils.WorldDataUtils", "format_ldraw"),
]


class TestImports(unittest.TestCase):
    def test_imports(self):
        """
        Dynamically test importing all modules and classes in 'text2brick'.
        """
        for module_name, class_or_func in imports_to_test:
            with self.subTest(module=module_name, member=class_or_func):
                try:
                    # Import the module
                    module = importlib.import_module(module_name)
                    # Access the class or function
                    getattr(module, class_or_func)
                    print(f"Successfully imported {class_or_func} from {module_name}")
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}. Error: {e}")
                except AttributeError as e:
                    self.fail(f"Failed to access {class_or_func} in {module_name}. Error: {e}")


if __name__ == "__main__":
    unittest.main()
