from typing import Tuple
import networkx as nx
import torch_geometric
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data
import torch
import numpy as np
import copy
import random

from text2brick.models import GraphWorldAbstract


class GraphLegoWorldData(GraphWorldAbstract):
    """
    A class representing a LEGO world as a graph where each brick is a node.
    The graph structure is based on a 2D array (img), where each brick is represented by '1, 1'.
    The class includes methods for creating the graph, adding bricks, checking connections, and updating the graph's validity.
    """

    graph : nx.Graph
    
    world_dim : Tuple[int, int, int]


    def __init__(self, img: np.ndarray):
        super().__init__(img)
        self.graph = self._create_graph_from_table(img)
        self.world_dim = (img.shape[0], img.shape[1], 1) # (y, x, z)


    def _create_graph_from_table(self, table: np.ndarray) -> nx.Graph:
        """
        Converts a 2D mapping array to a graph representation of the LEGO world.
        
        Args:
            table (np.ndarray): 2D array where 1 represents a stud and 0 represents empty space.
        
        Returns:
            nx.Graph: A graph where nodes represent bricks and edges represent connectivity.
        """
        graph = nx.Graph()
        rows = table.shape[0]
        brick_id = 0

        for row, y in zip(reversed(table), range(rows)):
            x = 0
            while x < len(row) - 1:  # Ensure there's space for a pair of ones
                if row[x] == 1 and row[x+1] == 1:  # Check for consecutive ones
                    graph.add_node(brick_id, x=x, y=y, saved=False, validity=True if y == 0 else False)
                    brick_id += 1
                    x += 2  # Skip the next index since it is part of the current block
                else:
                    x += 1  # Move to the next element

        if graph.number_of_nodes() == 0:
            graph.add_node(0, x=-1, y=-1, saved=True, validity=True)
        else:
            for brick_id1, data1 in graph.nodes(data=True):
                for brick_id2, data2 in graph.nodes(data=True):
                    if brick_id1 >= brick_id2:
                        continue
                    if self._check_connection(data1, data2):
                        graph.add_edge(brick_id1, brick_id2)

            graph = self._propagate_brick_validity(graph)
            self._remove_invalids(graph)

        return graph
    

    def graph_to_table(self) -> np.ndarray:
        """
        Converts the graph representation of the LEGO world back into a 2D table.
        Each brick (node) is represented by '1's in the table.

        Returns:
            np.ndarray: A 2D array where '1' represents part of a brick and '0' represents empty space.
        """
        table = np.zeros((self.world_dim[0], self.world_dim[1]), dtype=int)
        if self.is_world_empty():
            return table

        # Populate the table based on the brick positions in the graph
        for _, data in self.graph.nodes(data=True):
            x, y = data['x'], data['y']
            adjusted_y = self.world_dim[0] - 1 - y  # Adjust 'y' so that 0 corresponds to the last row
            table[adjusted_y, x] = 1
            if x + 1 < self.world_dim[1]:  # Check bounds to avoid index errors
                table[adjusted_y, x + 1] = 1

        return table
    

    def get_brick_at_edge(self):
        """
        Retrieve the brick at the edge (only one connection) or the brick at the highest y-coordinate.
        """
        # Nodes with degree 1 (edge nodes)
        edge_bricks = [node for node in self.graph.nodes if self.graph.degree[node] == 1 and self.graph.nodes[node].get('y', False) != 0]
       
        if edge_bricks:
            # Retrieve full node data and find the one with the highest y
            edge_bricks_data = [self.graph.nodes[node] for node in edge_bricks]
            return max(edge_bricks_data, key=lambda node: node["y"])

        # If no edge bricks, find articulation points in the graph
        articulation_points = set(nx.articulation_points(self.graph))
        
        # Consider all nodes and filter out the articulation points
        non_articulation_bricks = [self.graph.nodes[node] for node in self.graph.nodes if node not in articulation_points]
        
        # If there are non-articulation bricks, return the one with the highest y-coordinate
        if non_articulation_bricks:
            return max(non_articulation_bricks, key=lambda node: node["y"])

        print("No brick at edge", self.graph_to_table())


    def add_brick(self, x: int, y: int, debug: bool = False) -> bool:
        """
        Adds a new brick to the graph at the specified (x, y) coordinates, if possible.
        
        This method checks for:
        1. Overlap: Ensures no brick already exists at the given position.
        2. Boundaries: Ensures the brick is placed within the dimensions of the LEGO world.
        3. Connections: Updates the graph by connecting the new brick to valid neighbors.
        4. Validity propagation: Ensures the new graph remains structurally valid.

        If adding the brick invalidates part of the structure (e.g., it causes unsupported bricks to exist),
        the brick is removed, and the operation is considered unsuccessful.

        Args:
            x (int): The x-coordinate of the new brick's starting position.
            y (int): The y-coordinate of the new brick's position.
        
        Returns:
            bool: 
                - True if the brick was added successfully.
                - False if the brick couldn't be added due to overlap, out-of-bounds placement, 
                or causing the structure to become invalid.
        """
        if self.is_world_empty():
            self.graph.remove_node(0)

        if self._check_overlap(x, y):
            if debug:
                print(f"Overlap: A brick already exists at ({x}, {y})")
            return False
        if self._check_world_dimension(x, y):
            if debug:
                print(f"Outside: Brick at ({x}, {y}) is out of the world")
            return False

        brick_id = len(self.graph.nodes) 
        validity = True if y == 0 else False
        
        self.graph.add_node(brick_id, x=x, y=y, saved=False, validity=validity)
        
        # Update the edges (connections) between the new brick and its neighbors
        for neighbor_id, data in self.graph.nodes(data=True):
            if self._check_connection(self.graph.nodes[brick_id], data):
                self.graph.add_edge(brick_id, neighbor_id)

        self._propagate_brick_validity(self.graph)

        if self._remove_invalids(self.graph):
            if debug:
                print(f"Invalid: Brick at ({x}, {y})")
            return False

        if debug:
            print(f"Added brick at ({x}, {y})")
        return True
    

    def remove_brick(self, x: int, y: int, debug: bool = False) -> bool:
        """
        Removes a brick located at the specified (x, y) position from the graph.
        After removal, it propagates validity and removes any bricks that become invalid.

        Args:
            x (int): The x-coordinate of the brick to remove.
            y (int): The y-coordinate of the brick to remove.

        Returns:
            bool: True if the brick was successfully removed, False if no brick existed at the position.
        """
        if self.is_world_empty():
            if debug:
                print("No brick to remove, graph is empty")
            return False

        for node, data in self.graph.nodes(data=True):
            if data['x'] == x and data['y'] == y:
                self.graph.remove_nodes_from([node])
                self._propagate_brick_validity(self.graph)
                self._remove_invalids(self.graph)
                if self._remove_disconnected_subgraphs(debug=debug):
                    res = False
                if self.nodes_num() == 0:
                    self._empty_world()
                res = True
                if debug:
                    print(f"Removed brick at ({x}, {y})")
                return res

        if debug:
            print(f"No brick found at ({x}, {y}) to remove.")
        
        return False
    

    def _validity_mask(self) -> Tuple[np.array, np.array]:
        """
        Generates a validity mask for all positions in the world where a brick can be added.

        This method returns a 2D numpy array where each entry is:
        - 1 if a brick can be added at that position.
        - 0 if a brick cannot be added at that position.

        Returns:
            np.ndarray: A 2D array representing valid positions for brick placement.
        """
        mask = np.zeros((self.world_dim[0], self.world_dim[1]), dtype=int)
        min_x, max_x, _, max_y = self._min_max_x_y()
        placed_bricks = self._placed_brick()

        for y in range(self.world_dim[0]):
            for x in range(self.world_dim[1]):
                
                # Don't process when x or y is too far from the build
                if (x + 2 <= min_x or x - 2 >= max_x and y > 0) or y - 2 >= max_y:
                    continue

                # Skip if there is already a brick or the position is out of bounds
                if (x, y) in placed_bricks or self._check_world_dimension(x, y):
                    continue

                # Temporarily add the brick to check validity
                brick_id = len(self.graph.nodes)  # Get new brick ID
                self.graph.add_node(brick_id, x=x, y=y, saved=False, validity=True if y == 0 else False)

                # Update the graph by adding edges between the new brick and its neighbors
                for neighbor_id, data in self.graph.nodes(data=True):
                    if self._check_connection(self.graph.nodes[brick_id], data):
                        self.graph.add_edge(brick_id, neighbor_id)

                # Propagate validity after adding the brick
                self._propagate_brick_validity(self.graph)

                # Check if the brick placement keeps the structure valid
                if not self._remove_invalids(self.graph):
                    # If the graph is still valid, mark the position as valid for placement
                    mask[self.world_dim[0] - 1 - y, x] = 1

                # Safely remove the temporarily added brick
                if brick_id in self.graph:
                    self.graph.remove_node(brick_id)
        
        valid_coord = np.column_stack(np.where(mask == 1))
        valid_coord[:, 0] = mask.shape[0] - valid_coord[:, 0] - 1

        return mask, valid_coord[:, [1, 0]]

    
    def random_invalid_position(self, increase_dim: int = 0) -> dict: # To use for dataset generator
        """
        Generate a random invalid position within the world dimensions.

        This function repeatedly generates random coordinates and checks if they
        are invalid based on certain conditions (overlap, world dimension, and 
        connection to other nodes). An invalid position is returned once found.

        Returns:
            dict: A dictionary with 'x' and 'y' keys representing the invalid position.
        """
        placed_bricks = self._placed_brick()

        while True:
            coord = {
                'x': random.randint(0 - increase_dim, self.world_dim[1] + increase_dim),
                'y': random.randint(0 - increase_dim, self.world_dim[0] + increase_dim)
            }

            if ((coord['x'], coord['y'] not in placed_bricks)
                or not self._check_world_dimension(coord['x'], coord['y'])):
                return coord

            if coord['y'] == 0:
                continue

            for _, data in self.get_nodes():
                if self._check_connection(data, coord):
                    continue
            return coord
        
    
    def not_matching_pos(self, target: np.array) -> dict:
        """
        Finds the coordinates of a valid position that does not match the target array at a random index.
        
        This function identifies positions in the world where a brick can be placed (valid positions) 
        but do not match the target array (i.e., the target array has a value of 0 at those positions). 
        It then randomly selects one of these positions and returns its coordinates.

        Args:
            target (np.array): The target array to compare against. It contains 0s and 1s, where 0 represents 
                                an invalid position (or a mismatch) and 1 represents a valid position (matching).

        Returns:
            dict: A dictionary containing the 'x' and 'y' coordinates of a randomly selected non-matching position.
                - 'x': The x-coordinate of the position.
                - 'y': The y-coordinate of the position (adjusted for the world coordinate system).
        """
        if target.shape != self.world_dim[:2]:
            raise ValueError("Target dimensions doesn't match the world dimensions.")
        
        valid_mask, _ = self._validity_mask()
        not_matching_pos = np.column_stack(np.where((valid_mask == 1) & (target == 0))) # Valid positions but not matching the target
        index = random.randint(0, not_matching_pos.shape[0])
        not_matching_pos = not_matching_pos[index - 1, :]

        return {'x': not_matching_pos[1],
                'y': target.shape[0] - not_matching_pos[0] - 1
                }
    

    def _check_connection(self, data1, data2) -> bool:
        """
        Checks whether two bricks are connected based on the following criteria:
        - The bricks are horizontally adjacent (width condition).
        - The bricks are vertically adjacent (height condition).

        Args:
            x1, y1 (int): Coordinates of the first brick.
            x2, y2 (int): Coordinates of the second brick.
        
        Returns:
            bool: True if the bricks are connected, False otherwise.
        """
        if (
            abs(data1['x'] - data2['x']) <= 1                  # Width condition: bricks are horizontally adjacent
            and abs(data1['y'] - data2['y']) == 1             # Height condition: bricks are vertically adjacent
        ):
            return True
        return False
    

    def _check_world_dimension(self, x: int, y: int) -> bool:
        """
        Checks if the given coordinates (x, y) are within the valid world dimensions.
        
        This function verifies if the provided x and y coordinates are within the boundaries 
        of the world. The valid dimensions are determined by the world dimensions, where x 
        should be between 0 and `self.world_dim[1] - 1`, and y should be between 0 and 
        `self.world_dim[0] - 1`.

        Args:
            x (int): The x-coordinate to check.
            y (int): The y-coordinate to check.

        Returns:
            bool: 
                - True if the coordinates are out of bounds (either x or y is outside the valid range).
                - False if the coordinates are within the valid world dimensions.
        """
        if(
            x + 1 >= self.world_dim[1] or x < 0
            or y >= self.world_dim[0] or y < 0
        ):
            return True
        return False
     

    def _check_overlap(self, x: int, y: int) -> bool:
        """
        Checks if a brick already exists at the given coordinates (x, y), or if its adjacent positions are occupied.
        
        Args:
            x (int): The x-coordinate of the brick.
            y (int): The y-coordinate of the brick.
        
        Returns:
            bool: True if there is overlap (another brick exists at or adjacent to the position), False otherwise.
        """
        for _, data in self.get_nodes():
            if (data['x'] == x or data['x'] + 1 == x or data['x'] == x + 1) and data['y'] == y:
                return True
        return False
    

    def _min_max_x_y(self):
        """ 
        Returns the minimum and maximum values of the 'x' and 'y' attributes in a NetworkX graph's nodes using NumPy.

        Parameters:
        - graph (networkx.Graph): The input graph.

        Returns:
        - tuple: (min_x, max_x, min_y, max_y)
        """
        attributes = np.array([(data['x'], data['y']) for _, data in self.get_nodes() if 'x' in data and 'y' in data])
        
        if attributes.size == 0:
            raise ValueError("No nodes contain both 'x' and 'y' attributes.")
        
        min_x, min_y = np.min(attributes, axis=0)
        max_x, max_y = np.max(attributes, axis=0)
        
        return min_x, max_x, min_y, max_y
    

    def _placed_brick(self) -> set:
        """
        Returns a set of coordinates where bricks are placed, including adjacent positions.
        
        This function collects the positions of bricks that are already placed in the graph, 
        as well as the adjacent positions (to the right) of each placed brick. The coordinates 
        are returned as a set of tuples, where each tuple represents a coordinate `(x, y)`.

        Returns:
            set: A set containing the coordinates of placed bricks and their adjacent positions.
        """
        placed_bricks = set()
        for _, data in self.get_nodes():
            placed_bricks.add((data['x'], data['y']))
            placed_bricks.add((data['x'] + 1, data['y']))

        return placed_bricks
    

    def graph_to_torch(self, deepcopy=False, keep_unique_edge=False, attrs_to_keep=['x', 'y']) -> torch_geometric.data.Data:
        
        if self.graph.number_of_nodes() == 0:
            # Create an empty Data object
            return Data(x=torch.empty((0, 2), dtype=torch.float), edge_index=torch.empty((2, 0), dtype=torch.long))
       
        graph_cpy = self.graph

        if deepcopy:
            graph_cpy = copy.deepcopy(self.graph)

        # Convert from NetworkX to PyTorch Geometric Data
        data = from_networkx(graph_cpy, group_node_attrs=attrs_to_keep)
        data.x = data.x.float()
        data.edge_index = data.edge_index.long()  

        for key in list(data.keys()):
            if key not in attrs_to_keep + ['edge_index']:
                delattr(data, key)

        if keep_unique_edge:
            sorted_edges = torch.sort(data.edge_index, dim=0)[0]  # Sort each edge [min, max]
            unique_edges = torch.unique(sorted_edges, dim=1)  # Keep only unique edges
            data.edge_index = unique_edges

        return data