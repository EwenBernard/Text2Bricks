from typing import List, Tuple
import networkx as nx
import torch_geometric
from torch_geometric.utils.convert import from_networkx
import numpy as np
from collections import deque


"""
A class representing a LEGO world as a graph where each brick is a node.
The graph structure is based on a 2D array (img), where each brick is represented by '1'.
The class includes methods for creating the graph, adding bricks, checking connections, and updating the graph's validity.
"""
class GraphLegoWorldData:
    graph : nx.Graph
    brick_dim : Tuple[int, int, int] = (2, 1, 2)
    world_dim : Tuple[int, int, int] = (10, 10, 1)

    def __init__(self, img: np.ndarray):
        self.graph = self._create_graph_from_table(img)
        self.world_dim = (img.shape[0], img.shape[1], 1)


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
                    graph.add_node(brick_id, x=x, y=y, validity=True if y == 0 else False)
                    brick_id += 1
                    x += 2  # Skip the next index since it is part of the current block
                else:
                    x += 1  # Move to the next element

        for brick_id1, data1 in graph.nodes(data=True):
            x1, y1 = data1['x'], data1['y']
            for brick_id2, data2 in graph.nodes(data=True):
                if brick_id1 >= brick_id2:
                    continue
                x2, y2 = data2['x'], data2['y']
                if self._check_connection(x1, y1, x2, y2):
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

        # Populate the table based on the brick positions in the graph
        for _, data in self.graph.nodes(data=True):
            x, y = data['x'], data['y']
            adjusted_y = self.world_dim[0] - 1 - y  # Adjust 'y' so that 0 corresponds to the last row
            table[adjusted_y, x] = 1
            if x + 1 < self.world_dim[1]:  # Check bounds to avoid index errors
                table[adjusted_y, x + 1] = 1

        return table
    

    def add_brick(self, x: int, y: int) -> bool:
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
        if self._check_overlap(x, y):
            print(f"Overlap: A brick already exists at ({x}, {y})")
            return False
        if x + 1 >= self.world_dim[0] or y >= self.world_dim[1]:
            print(f"Outside: Brick at ({x}, {y}) is out of the world")
            return False

        brick_id = len(self.graph.nodes) 
        self.graph.add_node(brick_id, x=x, y=y, validity=True if y == 0 else False)
        
        # Update the edges (connections) between the new brick and its neighbors
        for neighbor_id, data in self.graph.nodes(data=True):
            neighbor_x, neighbor_y = data['x'], data['y']
            if self._check_connection(x, y, neighbor_x, neighbor_y):
                self.graph.add_edge(brick_id, neighbor_id)

        self._propagate_brick_validity(self.graph)

        if self._remove_invalids(self.graph):
            print(f"Invalid: Brick at ({x}, {y})")
            return False

        return True
    

    def remove_brick(self, x: int, y: int) -> bool:
        """
        Removes a brick located at the specified (x, y) position from the graph.
        After removal, it propagates validity and removes any bricks that become invalid.

        Args:
            x (int): The x-coordinate of the brick to remove.
            y (int): The y-coordinate of the brick to remove.

        Returns:
            bool: True if the brick was successfully removed, False if no brick existed at the position.
        """
        # Find the node that matches the given (x, y) coordinates
        node_to_remove = None
        for node, data in self.graph.nodes(data=True):
            if data['x'] == x and data['y'] == y:
                node_to_remove = node
                break

        if not node_to_remove:
            print(f"No brick found at ({x}, {y}) to remove.")
            return False
        
        self._propagate_brick_validity(self.graph)
        self._remove_invalids(self.graph)

        return True


    def _propagate_brick_validity(self, graph: nx.Graph) -> nx.Graph:
        """
        Propagates the validity of a brick to all connected bricks.
        If a brick is valid, all bricks connected to it will also be marked valid.
        
        Args:
            graph (nx.Graph): The graph representing the LEGO structure with nodes and edges.
            
        Returns:
            nx.Graph: The updated graph where all connected bricks to a valid brick are marked as valid.
        """
        valid_nodes = [node for node in graph.nodes if graph.nodes[node].get('validity', False)]     

        if not valid_nodes:
            return graph

        # Propagate validity to all connected bricks using BFS
        visited_for_validity = set(valid_nodes)  # Start with already valid nodes
        queue = deque(valid_nodes)
        
        while queue:
            current_node = queue.popleft()
            # Check connected nodes
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited_for_validity:
                    visited_for_validity.add(neighbor)
                    # Mark this brick as valid
                    graph.nodes[neighbor]['validity'] = True
                    queue.append(neighbor)

        return graph
    

    def _check_connection(self, x1: int, y1: int, x2: int, y2: int) -> bool:
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
            abs(x1 - x2) < self.brick_dim[0]  # Width condition: bricks are horizontally adjacent
            and abs(y1 - y2) == 1             # Height condition: bricks are vertically adjacent
        ):
            return True
        return False
    
    
    def _remove_invalids(self, graph: nx.Graph) -> bool:
        """
        Removes all invalid nodes (bricks) from the graph.
        
        Args:
            graph (nx.Graph): The graph representing the LEGO structure.
        
        Returns:
            bool: True if any invalid nodes were removed, False otherwise.
        """
        invalid_nodes = [node for node, data in graph.nodes(data=True) if not data.get('validity', False)]
        
        if invalid_nodes:
            graph.remove_nodes_from(invalid_nodes)
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
        for _, data in self.graph.nodes(data=True):
            if (data['x'] == x or data['x'] + 1 == x) and data['y'] == y:
                return True
        return False
    

    def nodes_num(self):
        return self.graph.number_of_nodes()
    

    def edges_num(self):
        return self.graph.number_of_edges()
     

    def print_graph(self):
        print(f"Number of nodes: {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")
        for node, data in self.graph.nodes(data=True):
            print(f"Node {node}: {data}")
        for u, v, data in self.graph.edges(data=True):
            print(f"Edge ({u}, {v}): {data}")

    
    def graph_to_torch(self) -> torch_geometric.data.Data:
        return from_networkx(self.graph)
    

    def graph_to_np(self) -> np.ndarray:
        return nx.to_numpy_array(self.graph)