import torch
from torch.utils.data import Dataset
import numpy as np
import random

from text2brick.models import BrickRef
from text2brick.managers.world.SingleBrickLegoWorldManager import SingleBrickLegoWorldManager
from text2brick.dataset.dataset import MNISTDataset


class CustomDatasetGraph(Dataset):
    
    def __init__(self):
        self.mnist = MNISTDataset()
        self.length = 1

    def __len__(self):
        # Return the number of samples
        return self.length

    # def __getitem2__(self, index):
    #     array, image, label, idx = self.mnist.sample()
    #     return array, image

    def __getitem__(self, index):

        array, _, _, _ = self.mnist.sample()
        mapping_table = [[0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 1, 1]]

        brick_ref = BrickRef(file_id="3003.dat", name="2x2", color=15, h=1, w=2, d=2)    
        lego_world_ref = SingleBrickLegoWorldManager(
            table=mapping_table,
            brick_ref=brick_ref,
        )
        print(lego_world_ref.world_2_tensor())

        # TODO: Get the graphs from the lego world
        # TODO: Generate random index between 0 and len(graphs)
        # TODO: Remove random_index nodes and clear the conections to get a new graphs
        # TODO: Return the new graph and the last node removed

        random_index = random.randint(0, len(lego_world_ref.data.world))
        print(random_index)

        i = 0
        for brick in reversed(lego_world_ref.data.world):
            if i >= random_index or len(lego_world_ref.data.world) == 0:
                res = lego_world_ref.world_2_tensor()
                print(res)
                return res
            v = lego_world_ref.remove_brick(brick)
            print(v)
            i += 1
        
        return res
            

        #nodes, _ = lego_world_ref.world_2_tensor()
        # add_later = []
        # default_graph = (torch.empty((0, 2), dtype=torch.int), torch.empty((0, 2), dtype=torch.int))
        # random_index = random.randint(0, len(nodes))

        # if random_index == len(nodes):
        #     return nodes


        # lego_world_cons = SingleBrickLegoWorldManager(
        #     table=np.zeros((env_size, env_size), dtype=np.uint8).tolist(),
        #     brick_ref=brick_ref,
        # )

        # def try_add_node(node):
        #     if lego_world_cons.add_brick_from_coord(node[0].item(), node[1].item(), brick_ref):
        #         return lego_world_cons.world_2_tensor()
        #     return default_graph

        # random_index = random.randint(0, len(nodes))
        # #random_index = 2

        # if random_index == 0:
        #     return default_graph

        # i = 0
        # while i < random_index or add_later:
        #     current_node = None

        #     if i < len(nodes):
        #         current_node = nodes[-(i + 1):][0]

        #     node_to_process = current_node if current_node is not None else add_later.pop(0)
        #     graph = try_add_node(node_to_process)

        #     if graph is not None and graph[0].shape[0] > 0:
        #         if i == random_index:
        #             return graph
        #     else:
        #         if current_node is not None:
        #             add_later.append(current_node)
        #         else:
        #             add_later.append(node_to_process)

        #     if current_node is not None:
        #         i += 1