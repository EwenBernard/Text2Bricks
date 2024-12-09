from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from text2brick.models.Brick import Brick, BrickRef
from text2brick.models.LegoWorldData import AbstractLegoWorldData
import logging

class AbstractLegoWorldManager(ABC):
    def __init__(self, table: Optional[List[List[int]]] = [], world_dimension: Tuple[int, int, int]= (10, 10, 1), **kwargs) -> None:
        pass 

    def add_brick_to_world(self, brick: Brick) -> None:
        """
        Add a brick to the world and update connections.

        Args:
            brick (Brick): The brick to add.
        """
        if self.check_brick_validity(brick):
            self.data.world.append(brick)
            self.add_brick_connection(brick)

            logging.debug(f"Added brick {brick.brick_id} to the world.")
            return True
        
        logging.debug(f"Brick {brick.brick_id} is invalid and was not added to the world.")
        return False
    

    def remove_brick_from_world(self, brick: Brick) -> None:
        """
        Remove a brick from the world and update connections.

        Args:
            brick (Brick): The brick to remove.
        """
        if brick in self.data.world:

            for other_brick in self.data.world:
                if brick in other_brick.connected_to:
                    other_brick.connected_to.remove(brick)

            self.data.world.remove(brick)
            self.valid_bricks.remove(brick.brick_id)

            logging.debug(f"Removed brick {brick.brick_id} from the world.")
            return True
        
        logging.debug(f"Brick {brick.brick_id} was not found in the world.")
        return False


    def add_brick_connection(self, brick: Brick) -> None:
        """
        Add a brick to the world and update connections.

        Args:
            brick (Brick): The brick to add.
        """
        for other_brick in self.data.world: 
            if brick != other_brick:
                brick.add_connection(other_brick)


    def _init_bricks_connections(self) -> None:
        """
        Initializes the connections between bricks in the world.

        Args:
            world (list of Brick): List of all bricks.
        """
        for brick in self.data.world:
           self.add_brick_connection(brick)

    
    def check_brick_validity(self, brick: Brick) -> bool:
        """
        Traverse the graph starting from a brick and mark connected bricks as valid.

        Args:
            brick (Brick): The starting brick for traversal.
        """

        # Brick is valid if y == 0 or connected to another valid brick
        if brick.y == 0 or any(conn.brick_id in self.valid_bricks for conn in brick.connected_to):
            self.valid_bricks.add(brick.brick_id)
            return True
        return False
    

    def init_brick_validity(self, brick: Brick) -> bool:
        """
        Traverse the graph starting from a brick and mark connected bricks as valid.

        Args:
            brick (Brick): The starting brick for traversal.
        """
        # If the brick has already been visited, return early
        if brick.brick_id in self.valid_bricks:
            return True
        
        # Brick is valid if y == 0 or connected to another valid brick
        if brick.y == 0 or any(conn.brick_id in self.valid_bricks for conn in brick.connected_to):
            self.valid_bricks.add(brick.brick_id)
            # Recursively check all connected bricks
            for conn in brick.connected_to:
                self.check_brick_validity(conn)
            return True
        
        return False


    def _init_world_validity(self, remove_illegal_bricks=True, return_illegal_bricks=False) -> Optional[List[Tuple[Brick, str]]]:
        """
        Check for illegal bricks in the world using a graph-based method.

        Args:
            world (list of Brick): All bricks in the world.

        Returns:
            Optional list of tuple: Each tuple contains the illegal brick and the reason.
        """
        self.valid_bricks = set()
        illegal_bricks = []

        for brick in self.world:
           self.check_brick_validity(brick)

        if remove_illegal_bricks:
            logging.debug(f"Removing illegal bricks from the world.")
            self.remove_illegal_bricks()

        if return_illegal_bricks:
            for brick in self.data.world:
                if brick.brick_id not in self.valid_bricks:
                    if brick.y < 0:
                        illegal_bricks.append((brick, "Floating brick - no connections and below ground"))
                    elif brick.y > 0:
                        illegal_bricks.append((brick, "Brick above ground level"))
            logging.debug(f"Found {len(illegal_bricks)} illegal bricks in the world :\n{illegal_bricks}")
        
            return illegal_bricks


    def remove_illegal_bricks(self) -> None:
        """
        Remove illegal bricks from the world.

        Args:
            world (list of Brick): All bricks in the world.
        """
        for brick in self.data.world:
            if brick.brick_id not in self.valid_bricks:
                self.data.world.remove(brick)


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
    
