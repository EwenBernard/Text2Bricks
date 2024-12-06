from .Brick import Brick
from typing import List

class LegoWorld: 
    def __init__(self, table, brick_ref):
        self.world : List[Brick] = []
        self._table_2_world(table, brick_ref)
        self.initialize_connections()

    def _table_2_world(self, table, brick_ref):
        """
        Converts a 2D mapping array to LDRAW LEGO brick coordinates, minimizing the number of bricks.
        
        Args:
            mapping_array (list of list of int): 2D array where 1 represents a stud and 0 represents empty space.
        
        Returns:
            list of tuple: Each tuple contains brick type and (X, Y) coordinates of the brick's bottom-left corner.
        """
        brick_id = 0
        rows = len(table)
        cols = len(table[0]) if rows > 0 else 0
        visited = [[False] * cols for _ in range(rows)]
        
        for y, height_multiplier in zip(range(rows), reversed(range(rows))):
            for x in range(cols):
                if table[y][x] == 1 and not visited[y][x]:
                    if x + 1 < cols and all(
                        table[y][x + dx] == 1 and not visited[y][x + dx]
                        for dx in range(2)
                    ):
                        brick_id += 1
                        self.world.append(Brick(x=x * brick_ref.w, y=-height_multiplier * brick_ref.h, z=0, brick_ref=brick_ref, brick_id=brick_id))
                        for dx in range(2):
                            visited[y][x + dx] = True

    def initialize_connections(self):
        """
        Initializes the connections between bricks in the world.

        Args:
            world (list of Brick): List of all bricks.
        """
        for brick in self.world:
            for other_brick in self.world:
                if brick != other_brick:
                    brick.add_connection(other_brick)


    def check_illegal_bricks(self):
        """
        Check for illegal bricks in the world using a graph-based method.

        Args:
            world (list of Brick): All bricks in the world.

        Returns:
            list of tuple: Each tuple contains the illegal brick and the reason.
        """
        visited = set()
        valid_bricks = set()  # Set of valid bricks (either directly on the ground or connected to valid ones)
        illegal_bricks = []

        def check_brick_validity(brick):
            """
            Traverse the graph starting from a brick and mark connected bricks as valid.

            Args:
                brick (Brick): The starting brick for traversal.
            """
            if brick.brick_id in visited:
                return
            visited.add(brick.brick_id)

            # Brick is valid if y == 0 or connected to another valid brick
            if brick.y == 0 or any(conn.brick_id in valid_bricks for conn in brick.connected_to):
                valid_bricks.add(brick.brick_id)
                for conn in brick.connected_to:
                    check_brick_validity(conn)

        # Initialize by checking all ground-level bricks
        for brick in self.world:
            if brick.y == 0:
                check_brick_validity(brick)

        # Check remaining bricks
        for brick in self.world:
            if brick.brick_id not in valid_bricks:
                if brick.y < 0 and not brick.connected_to:
                    illegal_bricks.append((brick, "Floating brick - no connections and below ground"))
                elif brick.y > 0:
                    illegal_bricks.append((brick, "Brick above ground level"))

        return illegal_bricks

    def _format_ldraw(self):
        ldr_line = []
        for brick in self.world:
            ldr_line.append(f"1 {brick.brick_ref.color} {brick.x} {brick.y} {brick.z} 1 0 0 0 1 0 0 0 1 {brick.brick_ref.file_id}")
        return ldr_line

    def save_ldr(self, filename):
        data = self._format_ldraw(self.world)
        with open(filename + ".ldr", "w") as file:
            for line in data:
                file.write(line + "\n")
    
    def __str__(self):
        return f"Piece Count : {len(self.world)}\n" + "\n".join([brick.coords() for brick in self.world])

    def str_full_infos(self):
        return "\n".join([str(brick) for brick in self.world])