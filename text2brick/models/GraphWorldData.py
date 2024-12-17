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

    def _create_graph_from_table(self, table: np.ndarray) -> nx.Graph:
        """
        Converts a 2D mapping array to a graph representation of the LEGO world.
        
        Args:
            table (np.ndarray): 2D array where 1 represents a stud and 0 represents empty space.
        
        Returns:
            nx.Graph: A graph where nodes represent bricks and edges represent connectivity.
        """
        rows, cols = table.shape
        graph = nx.Graph()
        
        if table.size == 0:
            return graph

        brick_id = 0
        visited = [[False] * cols for _ in range(rows)]
        node_positions = {}

        # Create nodes
        for y, height_multiplier in zip(range(rows), reversed(range(rows))):
            for x in range(cols):
                if table[y][x] == 1 and not visited[y][x]:
                    if x + self.brick_dim[0] <= cols and all(
                        table[y][x + dx] == 1 and not visited[y][x + dx]
                        for dx in range(self.brick_dim[0])):
                        
                        # Mark the brick position and create a node
                        brick_validity = True if height_multiplier == 0 else False
                        graph.add_node(brick_id, x=x, y=height_multiplier, brick_validity=brick_validity)
                        node_positions[brick_id] = (x, height_multiplier)

                        # Mark these positions as visited
                        for dx in range(self.brick_dim[0]):
                            visited[y][x + dx] = True
                        brick_id += 1

        # Create edges
        for node1, (x1, y1) in node_positions.items():
            for node2, (x2, y2) in node_positions.items():
                if node1 != node2:
                    if (
                        abs(x1 - x2) < self.brick_dim[0]  # Width condition
                        and abs(y1 - y2) == 1            # Height adjacency
                    ):
                        graph.add_edge(node1, node2)

        print(graph.nodes)

        return self._propagate_brick_validity(graph)


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
        valid_nodes = [node for node in graph.nodes if graph.nodes[node].get('brick_validity', False)]
        
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
                    graph.nodes[neighbor]['brick_validity'] = True
                    queue.append(neighbor)
        
        return graph

    
    def graph_to_torch(self) -> torch_geometric.data.Data:
        "converts the graph to torch_geometric.data.Data"
        return from_networkx(self.graph)
    
    def graph_to_np(self) -> np.ndarray:
        "converts the graph to numpy array"
        return nx.to_numpy_array(self.graph)


    
    
