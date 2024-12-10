from enum import Enum

class RemoveBrickBehaviorEnum(Enum):
    REMOVE_BRICK_ONLY = "REMOVE_BRICK_ONLY"  # Remove only the brick, keep the rest even if illegal
    REMOVE_AND_CLEAN = "REMOVE_AND_CLEAN"  # Remove the brick and any new illegal bricks
    SKIP_IF_ILLEGAL = "SKIP_IF_ILLEGAL"  # Don't remove the brick if it induces new illegal bricks

class BrickGetterEnum(Enum):
    ID = "ID" # brick_id
    COORDS = "COORDS" # (x, y, z)