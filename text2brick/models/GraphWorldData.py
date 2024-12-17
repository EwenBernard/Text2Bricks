from pydantic import BaseModel, Field
from typing import List, Tuple
import networkx as nx
import torch_geometric
from torch_geometric.utils.convert import from_networkx
import numpy as np


class GraphLegoWorldData(BaseModel):
    graph : nx.Graph = Field(..., description="Graph representation of the Lego world")
    brick_dim : Tuple[int, int, int] = (2, 1, 2)
    world_dim : Tuple[int, int, int] = (10, 10, 1)
    
    def graph_to_torch(self) -> torch_geometric.data.Data:
        "converts the graph to torch_geometric.data.Data"
        return from_networkx(self.graph)
    
    def graph_to_np(self) -> np.ndarray:
        "converts the graph to numpy array"
        return nx.to_numpy_array(self.graph)


    
    
