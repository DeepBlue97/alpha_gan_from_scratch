from turtle import forward
import torch
import torch.nn as nn


class VanillaModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.layer_1 = nn.Linear(3, 8)
        self.layer_2 = nn.Linear(8, 2)


    def forward(self, x):
        
        x1 = self.layer_1(x)
        out = self.layer_1(x1)

        return out
