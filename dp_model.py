from turtle import forward
import torch
import torch.nn as nn


class VanillaModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.layer_1 = nn.Linear(3, 8)
        self.layer_2 = nn.Linear(8, 2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.sigmoid(x)
        out = self.softmax(x)
        
        return out
