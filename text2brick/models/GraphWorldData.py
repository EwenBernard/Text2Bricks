from typing import Tuple
import networkx as nx
import torch_geometric
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data
import torch
import numpy as np
from collections import deque
import copy

from text2brick.models import BRICK_UNIT


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
        if self._is_graph_empty():
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
        
        # # If no edge bricks, consider all nodes and retrieve the one with the highest y
        # all_bricks_data = [self.graph.nodes[node] for node in self.graph.nodes]
        # return max(all_bricks_data, key=lambda node: node["y"])
        # If no edge bricks, find articulation points in the graph
        articulation_points = set(nx.articulation_points(self.graph))
        
        # Consider all nodes and filter out the articulation points
        non_articulation_bricks = [self.graph.nodes[node] for node in self.graph.nodes if node not in articulation_points]
        
        # If there are non-articulation bricks, return the one with the highest y-coordinate
        if non_articulation_bricks:
            return max(non_articulation_bricks, key=lambda node: node["y"])


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
        if self._is_graph_empty():
            self.graph.remove_node(0)

        if self._check_overlap(x, y):
            print(f"Overlap: A brick already exists at ({x}, {y})")
            return False
        if x + 1 >= self.world_dim[1] or y >= self.world_dim[0]:
            print(f"Outside: Brick at ({x}, {y}) is out of the world")
            return False

        brick_id = len(self.graph.nodes) 
        self.graph.add_node(brick_id, x=x, y=y, saved=False, validity=True if y == 0 else False)
        
        # Update the edges (connections) between the new brick and its neighbors
        for neighbor_id, data in self.graph.nodes(data=True):
            if self._check_connection(self.graph.nodes[brick_id], data):
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
        if self._is_graph_empty():
            print("No brick to remove, graph is empty")
            return False

        for node, data in self.graph.nodes(data=True):
            if data['x'] == x and data['y'] == y:
                self.graph.remove_nodes_from([node])
                self._propagate_brick_validity(self.graph)
                self._remove_invalids(self.graph)
                if self._remove_disconnected_subgraphs():
                    res = False
                if self.nodes_num() == 0:
                    self._empty_world()
                res = True
                return res

        print(f"No brick found at ({x}, {y}) to remove.")
        return False


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
        # print('Valid nodes:', valid_nodes)
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
            abs(data1['x'] - data2['x']) < self.brick_dim[0]  # Width condition: bricks are horizontally adjacent
            and abs(data1['y'] - data2['y']) == 1             # Height condition: bricks are vertically adjacent
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
            if (data['x'] == x or data['x'] + 1 == x or data['x'] == x + 1) and data['y'] == y:
                return True
        return False
    

    def _remove_disconnected_subgraphs(self):
        # Find all connected components in the graph
        connected_components = list(nx.connected_components(self.graph))
        res = False
        # Loop through each connected component
        for component in connected_components:
            # Check if any node in the component has 'y == 0'
            has_ground = any(self.graph.nodes[node].get('y') == 0 for node in component)
            
            # If the component does not have any node with 'y == 0', remove it
            if not has_ground:
                print(self.graph_to_table())
            
                self.graph.remove_nodes_from(component)
                print(self.graph_to_table())
                print(f"Removed disconnected subgraph: {component}")
                res = True
        return res

    

    def _empty_world(self):
        self.graph.add_node(0, x=-1, y=-1, saved=True, validity=True)


    def _is_graph_empty(self):
        if self.nodes_num() == 1:
            node = list(self.get_nodes())[0]
            if node[1].get('x') == -1 and node[1].get('y') == -1:
                return True
        return False
    

    def nodes_num(self):
        return self.graph.number_of_nodes()
    

    def edges_num(self):
        return self.graph.number_of_edges()
    

    def get_nodes(self):
        return self.graph.nodes(data=True)
    

    def get_edges(self):
        return self.graph.edges(data=True)
     

    def print_graph(self):
        print(f"Number of nodes: {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")
        for node, data in self.graph.nodes(data=True):
            print(f"Node {node}: {data}")
        for u, v, data in self.graph.edges(data=True):
            print(f"Edge ({u}, {v}): {data}")


    def subgraph(self, nodes_num):
        if nodes_num > self.graph.number_of_nodes():
            raise ValueError(f"Requested {nodes_num} nodes, but the graph only has {self.graph.number_of_nodes()} nodes.")

        # Select the specified number of nodes
        selected_nodes = list(self.graph.nodes)[:nodes_num]
        subgraph = self.graph.subgraph(selected_nodes).copy()

        return subgraph
    

    def graph_to_torch(self, deepcopy=True, keep_unique_edge=False, attrs_to_keep=['x', 'y']) -> torch_geometric.data.Data:
        
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
    

    def torch_to_graph(self, data: torch_geometric.data.Data) -> nx.Graph:
        return data.to_networkx()
    

    def graph_to_np(self) -> np.ndarray:
        return nx.to_numpy_array(self.graph)
    

    def save_as_ldraw(self, filename: str = "test_ldr", delete_all: bool = False):
        # Open the file in write mode if delete_all is True, else in append mode
        mode = "w" if delete_all else "a"
        with open(filename + ".ldr", mode) as file:
            for _, data in self.get_nodes():
                if not data['saved']:
                    # Create the LDraw line for the part
                    ldr_line = (
                        f"1 15 {data['x'] * BRICK_UNIT.W} {data['y'] * BRICK_UNIT.H} 1 "
                        f"1 0 0 0 1 0 0 0 1 3003.dat"
                    )
                    # Write the line to the file
                    file.write(ldr_line + "\n")
                    # Mark the part as saved
                    data['saved'] = True
