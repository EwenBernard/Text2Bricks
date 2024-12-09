from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Tuple, Set
from text2brick.models.Brick import Brick, BrickRef


class AbstractLegoWorldData(BaseModel, ABC):
    world : List[Brick] = Field(default_factory=list, description="List of bricks in the world")
    valid_bricks : Set[int] = Field(default_factory=set, description="Set of valid brick types")
    dimensions : Tuple[int, int, int] = Field(..., description="Dimensions of the Lego world (x, y, z)")

    def __str__(self) -> str:
        return f"Piece Count : {len(self.world)}\n" + "\n".join([brick.coords() for brick in self.world])

    def str_full_infos(self) -> str:
        return "Lego World Bricks\n" + "\n".join([str(brick) for brick in self.world])


class SingleBrickLegoWorldData(AbstractLegoWorldData): 
    brick_ref : BrickRef = Field(..., description="Reference to a specific brick")

    
