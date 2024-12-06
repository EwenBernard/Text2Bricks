from pydantic import BaseModel
from typing import List


class BrickRef(BaseModel):
    file_id: str
    name: str
    color: int
    h: int
    w: int
    d: int = 0

    def __str__(self):
        return f"ID: {self.file_id} | Name: {self.name} | Color: {self.color} | h:{self.h} | w:{self.w} | d:{self.d}"

    def shape(self):
        return (self.h, self.w, self.d)
    

class Brick(BaseModel):
    brick_id : int
    x: int
    y: int
    z: int
    brick_ref: BrickRef
    connected_to: List[BrickRef] = []

    def __str__(self):
        return f"id: {self.brick_id} x: {self.x} | y: {self.y} | z: {self.z} | {self.brick_ref} | Connected to: {[brick.brick_id for brick in self.connected_to]}"
    
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
    
    def shape(self):
        return self.brick_ref.shape()
    
    def coords(self):
        return f"x: {self.x} | y: {self.y} | z: {self.z}"
    