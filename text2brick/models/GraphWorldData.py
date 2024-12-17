from typing import List, Tuple
import networkx as nx
import torch_geometric
from torch_geometric.utils.convert import from_networkx
import numpy as np
from collections import deque

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
        rows, cols = len(table), len(table[0])
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
                if (
                    abs(x1 - x2) < self.brick_dim[0]  # Width condition
                    and abs(y1 - y2) == 1         # Height adjacency condition
                ):
                    graph.add_edge(brick_id1, brick_id2)

        graph = self._propagate_brick_validity(graph)

        # remove invalid nodes
        invalid_nodes = [node for node, data in graph.nodes(data=True) if not data.get('validity', False)]
        graph.remove_nodes_from(invalid_nodes)

        return graph


    def _propagate_brick_validity(self, graph: nx.Graph) -> nx.Graph:
        """
        Propagates the validity of a brick to all connected bricks.
        If a brick is valid, all bricks connected to it will also be marked valid.
        
        Args:
            graph (nx.Graph): The graph representing the LEGO structure with nodes and edges.
            
        Returns:
            nx.Graph: The updated graph where all connected bricks to a valid brick are marked as valid.
        """

        # Step 1: Find all nodes that are valid (brick_validity=True)
        valid_nodes = [node for node in graph.nodes if graph.nodes[node].get('validity', False)]     

        # Step 2: If there are no valid bricks initially, return the graph as is
        if not valid_nodes:
            return graph

        # Step 3: Propagate validity to all connected bricks using BFS
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
    
    
    def add_brick(self, x, y):
        self.graph.add_node(x, y, True if y == 0 else False)
    
    def print_graph(self):
        for node, data in self.graph.nodes(data=True):
            print(f"Node {node}: {data}")

        for u, v, data in self.graph.edges(data=True):
            print(f"Edge ({u}, {v}): {data}")

    
    def graph_to_torch(self) -> torch_geometric.data.Data:
        "converts the graph to torch_geometric.data.Data"
        return from_networkx(self.graph)
    
    def graph_to_np(self) -> np.ndarray:
        "converts the graph to numpy array"
        return nx.to_numpy_array(self.graph)


    
    
