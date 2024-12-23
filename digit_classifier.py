import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Define the architecture here
        self.first_layer = nn.Linear(784, 512)
        self.relu_layer = nn.ReLU()
        self.dropout_layer = nn.Dropout(0.2)
        self.final_layer = nn.Linear(512, 10)
        self.sigmoid_layer = nn.Sigmoid()
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # Return the model's prediction to 4 decimal places
        images = self.first_layer(images)
        images = self.relu_layer(images)
        images = self.dropout_layer(images)
        images = self.final_layer(images)
        images = self.sigmoid_layer(images)

        return torch.round(images, decimals = 4)
    





#########Alternate Solution#######
import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.first_linear = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.projection = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        out = self.sigmoid(self.projection(self.dropout(self.relu(self.first_linear(images)))))
        return torch.round(out, decimals=4)

