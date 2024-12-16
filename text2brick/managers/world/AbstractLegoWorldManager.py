from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from text2brick.models import AbstractLegoWorldData, RemoveBrickBehaviorEnum, Brick, BrickRef, BrickGetterEnum
import logging
import torch

class AbstractLegoWorldManager(ABC):
    def __init__(self, table: Optional[List[List[int]]] = [], world_dimension: Tuple[int, int, int]= (10, 10, 1), **kwargs) -> None:
        self.data : AbstractLegoWorldData
        pass 


    def get_brick(self, identifier: Union[Tuple[int,int,int], int], lookup_type: BrickGetterEnum = BrickGetterEnum.COORDS) -> Brick:
        """
        Get a brick from the world using a specific getter.

        Args:
            by (BrickGetterEnum): The getter to use.

        Returns:
            Brick: The brick found.
        """
        if lookup_type == BrickGetterEnum.COORDS:
            for brick in self.data.world:
                if brick.x == identifier[0] and brick.y == identifier[1] and brick.z == identifier[2]:
                    return brick
        
        elif lookup_type == BrickGetterEnum.ID:
             for brick in self.data.world:
                if brick.brick_id == identifier:
                    return brick


    def add_brick(self, brick: Brick) -> None:
        """
        Add a brick to the world and update connections.

        Args:
            brick (Brick): The brick to add.
        """
        if not self._check_bricks_overlap_in_world(brick, self.data.world):
            self.add_brick_connection(self.data.world, brick)

            if self.check_brick_validity(brick):
                self.data.world.append(brick)
                logging.debug(f"Added brick {brick.brick_id} to the world.")
                return True
            
        logging.debug(f"Brick {brick.brick_id} is invalid and was not added to the world.")
        return False


    def add_brick_from_coord(self, x, y, brick_ref: BrickRef):
        """
        Adds a brick to the LEGO world using its coordinates.

        Args:
            x (int): The x-coordinate of the brick in the LEGO world.
            y (int): The y-coordinate of the brick in the LEGO world.
            brick_ref (BrickRef): Reference object containing brick properties (e.g., dimensions, color).

        Returns:
            bool: True if the brick was successfully added, False otherwise.
        """
        if self.data.world:
            id = self.data.world[-1].brick_id + 1 
        else:
            id = 0

        brick = Brick(brick_id=id, x=x, y=y, z=0, brick_ref=brick_ref)
        return self.add_brick(brick)


    def remove_brick(self, brick: Brick, rm_behavior: RemoveBrickBehaviorEnum = RemoveBrickBehaviorEnum.SKIP_IF_ILLEGAL) -> bool:
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
        temp_world = [b for b in self.data.world if b != brick]

        temp_world = AbstractLegoWorldData(world=temp_world, brick_ref=self.data.brick_ref, dimensions=self.data.dimensions)
        self._init_bricks_connections(temp_world)
        invalid_bricks = self._init_world_validity(temp_world, remove_illegal_bricks=False, return_illegal_bricks=True)

        if rm_behavior == RemoveBrickBehaviorEnum.SKIP_IF_ILLEGAL and invalid_bricks:
            logging.debug(f"Brick {brick.brick_id} cannot be removed as it makes the world invalid.")
            return False

        # Update other bricks connections and delete brick
        self._update_brick_connections(brick)
        self._delete_brick_from_data(brick)

        # Remove invalid bricks if specified
        if rm_behavior == RemoveBrickBehaviorEnum.REMOVE_AND_CLEAN:
            for brick in invalid_bricks:
                self._delete_brick_from_data(brick)

        logging.debug(f"Removed brick {brick.brick_id} from the world.")
        return True


    def add_brick_connection(self, world: List[Brick], brick: Brick) -> None:
        """
        Add a brick to the world and update connections.

        Args:
            brick (Brick): The brick to add.
        """
        for other_brick in world: 
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


    def _delete_brick_from_data(self, brick: Brick) -> None:
        """
        Remove all brick from the world.
        
        Args:
            brick: brick to remove.
        """
        self.data.world.remove(brick)
        self.data.valid_bricks.discard(brick.brick_id)
    

    def _init_brick_validity(self, data: AbstractLegoWorldData, brick: Brick, visited=None) -> bool:
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

        if brick.brick_id in data.valid_bricks:
            return True

        if brick.brick_id in visited:
            return False

        # Check validity for connected bricks
        visited.add(brick.brick_id)
        if brick.y == 0:
            data.valid_bricks.add(brick.brick_id)
            queue = [brick]
            while queue:
                current_brick = queue.pop(0)
                for conn in current_brick.connected_to:
                    if conn.brick_id in data.valid_bricks:
                        data.valid_bricks.add(current_brick.brick_id)
                    if conn.brick_id not in visited:
                        visited.add(conn.brick_id)
                        queue.append(conn)
            return True

        return False
    

    def _brick_overlap(self, brick1: Brick, brick2: Brick) -> bool:
        """
        Check if two bricks overlap based on their positions and dimensions.

        Args:
            brick1 (Brick): The first brick.
            brick2 (Brick): The second brick.

        Returns:
            bool: True if the bricks overlap, False otherwise.
        """
        overlap_x = (brick1.x < brick2.x + brick2.brick_ref.w) and (brick1.x + brick1.brick_ref.w > brick2.x)
        overlap_y = (brick1.y < brick2.y + brick2.brick_ref.h) and (brick1.y + brick1.brick_ref.h > brick2.y)
        overlap_z = (brick1.z < brick2.z + brick2.brick_ref.d) and (brick1.z + brick1.brick_ref.d > brick2.z)
        return overlap_x and overlap_y and overlap_z


    def _check_bricks_overlap_in_world(self, brick: Brick, world: List[Brick]) -> bool:
        """
        Check if a brick overlaps with any other bricks in the world.

        Args:
            brick (Brick): The brick to check.
            world (list of Brick): List of all bricks in the world.

        Returns:
            bool: True if the brick overlaps with any other bricks, False otherwise.
        """
        for other_brick in world:
            if self._brick_overlap(brick, other_brick):
                return True
        return False


    def _init_bricks_connections(self, data: AbstractLegoWorldData) -> None:
        """
        Initializes the connections between bricks in the world.

        Args:
            world (list of Brick): List of all bricks.
        """
        for brick in data.world:
           self.add_brick_connection(data.world, brick)


    def _init_world_validity(self, data: AbstractLegoWorldData, remove_illegal_bricks=True, return_illegal_bricks=False) -> Optional[List[Tuple[Brick, str]]]:
        """
        Check for illegal bricks in the world using a graph-based method.

        Args:
            world (list of Brick): All bricks in the world.

        Returns:
            Optional list of tuple: Each tuple contains the illegal brick and the reason.
        """
        
        for brick in data.world:
            self._init_brick_validity(data, brick)

        if remove_illegal_bricks:
            logging.debug(f"Removing illegal bricks from the world.") 
            for brick in data.world:
                if brick.brick_id not in data.valid_bricks:
                    self.remove_brick(brick)

        if return_illegal_bricks:
            illegal_bricks = []
            for brick in data.world:
                if brick.brick_id not in data.valid_bricks:
                    if brick.y > 0:
                        illegal_bricks.append((brick, "Floating brick - no connection or connected to illegal brick"))
                    elif brick.y < 0:
                        illegal_bricks.append((brick, "Brick above ground level"))
            logging.debug(f"Found {len(illegal_bricks)} illegal bricks in the world :\n{illegal_bricks}")
        
            return illegal_bricks
        
    @abstractmethod
    def _world_2_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts the world data to a tensor representation.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Tuple containing:
                - Tensor representation of the world data. shape(num_bricks, nb_features)
                - Edge index tensor representing the connections between bricks. shape(2, num_connections)
        """
        pass

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
    
    @abstractmethod
    def recreate_table_from_world(self, brick_world : AbstractLegoWorldData) -> List[List[int]]:
        """
        Converts a LEGO brick world back into a 2D mapping array.

        Args:
            brick_world (list of Brick): List of Brick objects representing the LEGO world.
            world_dimension (tuple): Dimensions of the world as (rows, cols, height). Defaults to (10, 10, 1).

        Returns:
            list of list of int: array where 1 represents a stud and 0 represents empty space.
        """