from pydantic import BaseModel, Field
from typing import List, Tuple
from networkx import Graph


class GraphLegoWorldData(BaseModel):
    graph : Graph = Field(..., description="Graph representation of the Lego world")
    brick_dim : Tuple[int, int, int] = (2, 1, 2)
    mapping_table : List[List[List[int]]] = Field(default_factory=list, description="Mapping table of the Lego world")

    def __str__(self) -> str:
        return f"Piece Count : {len(self.world)}\n" + "\n".join([brick.coords() for brick in self.world])

    def str_full_infos(self) -> str:
        return "Lego World Bricks\n" + "\n".join([str(brick) for brick in self.world])