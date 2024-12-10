from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from text2brick.models import AbstractLegoWorldData, RemoveBrickBehaviorEnum, Brick, BrickRef
import logging


class AbstractLegoWorldManager(ABC):
    def __init__(self, table: Optional[List[List[int]]] = [], world_dimension: Tuple[int, int, int]= (10, 10, 1), **kwargs) -> None:
        self.data : AbstractLegoWorldData
        pass 

    def add_brick_to_world(self, brick: Brick) -> None:
        """
        Add a brick to the world and update connections.

        Args:
            brick (Brick): The brick to add.
        """
        self.add_brick_connection(brick)
        if self.check_brick_validity(brick):
            self.data.world.append(brick)

            logging.debug(f"Added brick {brick.brick_id} to the world.")
            return True
        
        logging.debug(f"Brick {brick.brick_id} is invalid and was not added to the world.")
        return False
    

    def add_brick_to_world_from_coord(self, x, y, brick_ref: BrickRef):
        """
        Adds a brick to the LEGO world using its coordinates.

        Args:
            x (int): The x-coordinate of the brick in the LEGO world.
            y (int): The y-coordinate of the brick in the LEGO world.
            brick_ref (BrickRef): Reference object containing brick properties (e.g., dimensions, color).

        Returns:
            bool: True if the brick was successfully added, False otherwise.
        """
        if len(self.data.world) != 0:
            id = self.data.world[-1].brick_id + 1 
        else:
            id = 1 

        brick = Brick(brick_id=id, x=x, y=y, z=0, brick_ref=brick_ref)
        return self.add_brick_to_world(brick)



    def remove_brick_from_world(self, brick: Brick, rm_behavior: RemoveBrickBehaviorEnum = RemoveBrickBehaviorEnum.SKIP_IF_ILLEGAL) -> bool:
        """
        Remove a brick from the world and update connections.

        Args:
            brick (Brick): The brick to remove.
            rm_behavior (RemoveBrickBehaviorEnum): Behavior to follow when removing the brick.
        
        Returns:
            bool: True if the brick was successfully removed, False otherwise.
        """
        if brick not in self.data.world:
            logging.debug(f"Brick {brick.brick_id} was not found in the world.")
            return False

        # Create a temporary copy of the world for validation
        temp_world = self.data.model_copy()
        temp_world.world.remove(brick)

        # Check validity of the world after removing the brick
        invalid_bricks = [b for b in temp_world.world if not self._init_brick_validity(b)]
        
        if rm_behavior == RemoveBrickBehaviorEnum.SKIP_IF_ILLEGAL and invalid_bricks:
            logging.debug(f"Brick {brick.brick_id} cannot be removed as it makes the world invalid.")
            return False

        # Update connections
        self._update_brick_connections(brick)
        
        # Remove the brick
        self.data.world.remove(brick)
        self.data.valid_bricks.discard(brick.brick_id)

        # Remove invalid bricks if specified
        if rm_behavior == RemoveBrickBehaviorEnum.REMOVE_AND_CLEAN:
            self._remove_invalid_bricks(invalid_bricks)

        logging.debug(f"Removed brick {brick.brick_id} from the world.")
        return True


    def add_brick_connection(self, brick: Brick) -> None:
        """
        Add a brick to the world and update connections.

        Args:
            brick (Brick): The brick to add.
        """
        for other_brick in self.data.world: 
            if brick != other_brick:
                brick.add_connection(other_brick)

    
    def check_brick_validity(self, brick: Brick) -> bool:
        """
        Check if a brick is valid in the context of its connections.
        
        Args:
            brick (Brick): The brick to check.
        
        Returns:
            bool: True if the brick is valid, False otherwise.
        """
        if brick.y == 0 or any(conn.brick_id in self.data.valid_bricks for conn in brick.connected_to):
            self.data.valid_bricks.add(brick.brick_id)
            return True
        return False
    

    def _update_brick_connections(self, brick: Brick) -> None:
        """
        Update connections when a brick is removed from the world.
        
        Args:
            brick (Brick): The brick to update connections for.
        """
        for other_brick in self.data.world:
            if brick in other_brick.connected_to:
                other_brick.connected_to.remove(brick)


    def _remove_invalid_bricks(self, invalid_bricks: List[Brick]) -> None:
        """
        Remove all invalid bricks from the world.
        
        Args:
            invalid_bricks (list of Brick): List of invalid bricks to remove.
        """
        for brick in invalid_bricks:
            self.data.world.remove(brick)
            self.data.valid_bricks.discard(brick.brick_id)
    

    def _init_brick_validity(self, brick: Brick, visited=None) -> bool:
        """
        Initialize the validity of a brick and its connections using bread-first search.
        
        Args:
            brick (Brick): The starting brick.
            visited (set): A set to track visited bricks during traversal.
        
        Returns:
            bool: True if the brick is valid, False otherwise.
        """
        if visited is None:
            visited = set()

        # Early exit if already valid
        if brick.brick_id in self.data.valid_bricks:
            return True

        if brick in visited:
            return False

        # Check validity for connected bricks
        visited.add(brick)
        if brick.y == 0 or any(conn.brick_id in self.data.valid_bricks for conn in brick.connected_to):
            self.data.valid_bricks.add(brick.brick_id)
            queue = [brick]
            while queue:
                current_brick = queue.pop(0)
                for conn in current_brick.connected_to:
                    if conn not in visited:
                        visited.add(conn)
                        queue.append(conn)
            return True

        return False


    def _init_bricks_connections(self) -> None:
        """
        Initializes the connections between bricks in the world.

        Args:
            world (list of Brick): List of all bricks.
        """
        for brick in self.data.world:
           self.add_brick_connection(brick)


    def _init_world_validity(self, data: AbstractLegoWorldData, remove_illegal_bricks=True, return_illegal_bricks=False) -> Optional[List[Tuple[Brick, str]]]:
        """
        Check for illegal bricks in the world using a graph-based method.

        Args:
            world (list of Brick): All bricks in the world.

        Returns:
            Optional list of tuple: Each tuple contains the illegal brick and the reason.
        """
        
        for brick in data.world:
            self._init_brick_validity(brick)

        if remove_illegal_bricks:
            logging.debug(f"Removing illegal bricks from the world.") 
            for brick in data.world:
                if brick.brick_id not in data.valid_bricks:
                    self.remove_brick_from_world(brick)

        if return_illegal_bricks:
            illegal_bricks = set()
            for brick in data.world:
                if brick.brick_id not in data.valid_bricks:
                    if brick.y < 0:
                        illegal_bricks.add((brick, "Floating brick - no connection or connected to illegal brick"))
                    elif brick.y > 0:
                        illegal_bricks.add((brick, "Brick above ground level"))
            logging.debug(f"Found {len(illegal_bricks)} illegal bricks in the world :\n{illegal_bricks}")
        
            return illegal_bricks


    @abstractmethod
    def _create_world_from_table(self, table, world_dimension, **kwargs) -> AbstractLegoWorldData:
        """
        Converts a mapping array to LDRAW LEGO brick coordinates, minimizing the number of bricks.
        
        Args:
            mapping_array : array where 1 represents a stud and 0 represents empty space.
        
        Returns:
            AbstractLegoWorldData: Data class containing the world and brick reference.
        """
        pass
    
