import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.first_layer = nn.Linear(4,6)
        self.second_layer = nn.Linear(6,4)
        self.output = nn.Linear(4,2)
    
    def forward(self, x):
        x = self.first_layer(x)
        x = self.second_layer(x)
        x = self.output(x)
        return x
    

model = MyModel()

input_tensor = torch.randn(1,4)
output = model(input_tensor)
print(output)