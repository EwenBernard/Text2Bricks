from typing import List
from text2brick.models import Brick, BrickRef, SingleBrickLegoWorldData
from text2brick.managers.world.AbstractLegoWorldManager import AbstractLegoWorldManager
import logging

class SingleBrickLegoWorldManager(AbstractLegoWorldManager): 
    def __init__(self, table: List[List[int]], brick_ref: BrickRef = None, world_dimension = (10, 10, 1), remove_illegal_brick_init=True) -> None:
        super().__init__(table=table, world_dimension=world_dimension)
        self.data : SingleBrickLegoWorldData = self._create_world_from_table(table, brick_ref, world_dimension=world_dimension)

        if self.data.world:
            self._init_bricks_connections()
            self._init_world_validity(remove_illegal_bricks=remove_illegal_brick_init)

        logging.debug(f"Initialized Lego World Manager with {len(self.data.world)} bricks.")
        logging.debug(f"World dimensions : {self.data.dimensions}")
        logging.debug(f"Valid bricks : {self.data.valid_bricks}")
        logging.debug(f"World : {self.data.world}")


    def _create_world_from_table(self, table, brick_ref, world_dimension=(10, 10, 1)):
        """
        Converts a 2D mapping array to LDRAW LEGO brick coordinates, minimizing the number of bricks.
        
        Args:
            mapping_array (list of list of int): 2D array where 1 represents a stud and 0 represents empty space.
        
        Returns:
            list of tuple: Each tuple contains brick type and (X, Y) coordinates of the brick's bottom-left corner.
        """

        if not table:
            logging.warning(f"Empty table provided. Returning empty world with dimensions {world_dimension}")
            return SingleBrickLegoWorldData(world=[], brick_ref=brick_ref, dimensions=world_dimension)
        
        brick_id = 0
        rows = len(table)
        cols = len(table[0]) if rows > 0 else 0
        visited = [[False] * cols for _ in range(rows)]
        world = []
        
        for y, height_multiplier in zip(range(rows), reversed(range(rows))):
            for x in range(cols):
                if table[y][x] == 1 and not visited[y][x]:
                    if x + 1 < cols and all(
                        table[y][x + dx] == 1 and not visited[y][x + dx]
                        for dx in range(2)
                    ):
                        brick_id += 1
                        world.append(Brick(x=x * brick_ref.w / 2, y=-height_multiplier * brick_ref.h, z=0, brick_ref=brick_ref, brick_id=brick_id))
                        for dx in range(2):
                            visited[y][x + dx] = True
        
        return SingleBrickLegoWorldData(world=world, brick_ref=brick_ref, dimensions=(rows, cols, 1))