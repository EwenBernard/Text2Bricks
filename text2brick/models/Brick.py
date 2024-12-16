from pydantic import BaseModel, field_validator, ValidationError
from typing import List
from .Constants import BRICK_UNIT


class BrickRef(BaseModel):
    #TODO use it with lego unit instead of full lenght
    file_id: str
    name: str
    color: int
    h: int
    w: int
    d: int = 0

    def __str__(self):
        return f"ID: {self.file_id} | Name: {self.name} | Color: {self.color} | h:{self.h} | w:{self.w} | d:{self.d}"

    def __eq__(self, other):
        if isinstance(other, BrickRef):
            return (self.file_id, self.name, self.color, self.h, self.w, self.d) == (other.file_id, other.name, other.color, other.h, other.w, other.d)
        return False

    def __hash__(self):
        return hash((self.file_id, self.name, self.color, self.h, self.w, self.d))

    def shape(self):
        return (self.h, self.w, self.d)
    
    def convert_to_unit(self):
        return (self.h * BRICK_UNIT.H, self.w * BRICK_UNIT.W, self.d * BRICK_UNIT.D)
    

class Brick(BaseModel):
    #TODO change brickref to brick
    brick_id : int
    x: int
    y: int
    z: int
    brick_ref: BrickRef
    connected_to: List[BrickRef] = []

    # @field_validator("y")
    # def check_y_dimension(cls, v):
    #     if v > 0:
    #         raise ValidationError("The 'y' dimension must be inferior or equal to 0.")
    #     return v

    def __str__(self):
        return f"id: {self.brick_id} x: {self.x} | y: {self.y} | z: {self.z} | {self.brick_ref} | Connected to: {[brick.brick_id for brick in self.connected_to]}"
    
    def __eq__(self, other):
        if isinstance(other, Brick):
            return (self.brick_id, self.x, self.y, self.z, self.brick_ref) == (other.x, other.y, other.z, other.brick_ref)
        return False
    
    def is_connected_to(self, other_brick):
        """
        Checks if this brick is physically connected to another brick.

        Args:
            other_brick (Brick): The brick to check against.

        Returns:
            bool: True if connected, False otherwise.
        """

        if (
            abs(self.x - other_brick.x) < self.brick_ref.w and
            abs(self.z - other_brick.z) < self.brick_ref.d and
            (
                self.y == other_brick.y - other_brick.brick_ref.h or
                other_brick.y == self.y - self.brick_ref.h
            )
        ):
            #print(f"Connected: {self.brick_id} and {other_brick.brick_id}")
            return True
        return False

    def add_connection(self, other_brick):
        """
        Adds a connection to another brick if connected.

        Args:
            other_brick (Brick): The brick to potentially connect to.
        """
        if self.is_connected_to(other_brick):
            self.connected_to.append(other_brick)

    def get_xy_coords(self):
        return (self.x, self.y)
    
    def get_xyz_coords(self):
        return (self.x, self.y, self.z)
    
    def shape(self):
        return self.brick_ref.shape()
    
    def coords(self):
        return f"x: {self.x} | y: {self.y} | z: {self.z}"