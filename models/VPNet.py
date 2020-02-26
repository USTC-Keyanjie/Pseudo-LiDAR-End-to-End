import torch
import torch.nn as nn
import torch.nn.functional as F


class VPNet(nn.Module):
    def __init__(self):
        super(VPNet, self).__init__()

    def DBH_indice(self, depth_map):
        max_1 = torch.max(depth_map, 1)

        DBH_x = torch.max(max_1[0], 1)[1].squeeze()
        DBH_y = max_1[1].squeeze()[DBH_x]

        return DBH_x, DBH_y

    def forward(self, input_data):
        depth_map = input_data['depth_map']
        DBH_x, DBH_y = self.DBH_indice(depth_map)
        return DBH_x, DBH_y
