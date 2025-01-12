from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from typing import Tuple
from collections import deque
import torch_geometric
from torch_geometric.utils.convert import to_networkx

from text2brick.models import BRICK_UNIT


class GraphWorldAbstract(ABC):

    graph : nx.Graph
    world_dim : Tuple[int, int, int]

    def __init__(self, img: np.ndarray) -> None:
        super().__init__()

    @abstractmethod
    def _create_graph_from_table(self, table: np.ndarray) -> nx.Graph:
        pass

    @abstractmethod
    def add_brick(self, x: int, y: int, *args, **kwargs) -> bool:
        pass
    
    @abstractmethod
    def remove_brick(self, x: int, y: int, *args, **kwargs) -> bool:
        pass
    
    @abstractmethod
    def graph_to_torch(self, *args, **kwargs) -> torch_geometric.data.Data:
        pass


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
    

    def _remove_disconnected_subgraphs(self, debug: bool = True) -> None:
        """
        Removes disconnected subgraphs from the main graph. A subgraph is considered disconnected if 
        it does not contain any node with 'y == 0' (which represents ground level).
        
        Args:
            debug (bool, optional): If True, prints debug information about the graph before and after removal.
        
        Returns:
            bool: True if any disconnected subgraphs were removed, False otherwise.
        """
        # Find all connected components in the graph
        connected_components = list(nx.connected_components(self.graph))
        res = False
        # Loop through each connected component
        for component in connected_components:
            # Check if any node in the component has 'y == 0'
            has_ground = any(self.graph.nodes[node].get('y') == 0 for node in component)
            
            # If the component does not have any node with 'y == 0', remove it
            if not has_ground:

                self.graph.remove_nodes_from(component)
                if debug:
                    print(f"Removed disconnected subgraph: {component}")
                res = True
        return res
    

    def _empty_world(self) -> None:
        self.graph.add_node(0, x=-1, y=-1, saved=True, validity=True)


    def is_world_empty(self) -> bool:
        if self.nodes_num() == 1:
            node = list(self.get_nodes())[0]
            if node[1].get('x') == -1 and node[1].get('y') == -1:
                return True
        return False
    

    def nodes_num(self) -> int:
        return self.graph.number_of_nodes()
    

    def edges_num(self) -> int:
        return self.graph.number_of_edges()
    

    def get_nodes(self):
        return self.graph.nodes(data=True)
    

    def get_edges(self):
        return self.graph.edges(data=True)
     

    def print_graph(self) -> None:
        print(f"Number of nodes: {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")
        for node, data in self.graph.nodes(data=True):
            print(f"Node {node}: {data}")
        for u, v, data in self.graph.edges(data=True):
            print(f"Edge ({u}, {v}): {data}")



    def torch_to_graph(self, data: torch_geometric.data.Data) -> nx.Graph:
        return to_networkx(data)
    

    def graph_to_np(self) -> np.ndarray:
        return nx.to_numpy_array(self.graph)
    


    def subgraph(self, nodes_num: int) -> nx.Graph:
        """
        Creates and returns a subgraph containing a specified number of nodes from the original graph.

        Args:
            nodes_num (int): The number of nodes to include in the subgraph.

        Returns:
            nx.Graph: A subgraph containing the specified number of nodes.

        Raises:
            ValueError: If the requested number of nodes exceeds the total number of nodes in the graph.
        """
        if nodes_num > self.graph.number_of_nodes():
            raise ValueError(f"Requested {nodes_num} nodes, but the graph only has {self.graph.number_of_nodes()} nodes.")

        # Select the specified number of nodes
        selected_nodes = list(self.graph.nodes)[:nodes_num]
        subgraph = self.graph.subgraph(selected_nodes).copy()

        return subgraph


    def save_as_ldraw(self, filename: str = "test_ldr", delete_all: bool = False) -> None:
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