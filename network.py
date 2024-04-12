import torch
from torch import nn

# NOTE: This will be the network architecture. 

class Net(nn.Module):
    def __init__(self, nClasses):
        super().__init__()

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(3, 9, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(9, 16, kernel_size=3, padding=1)

        self.fc1 = (16*16*16, 512)
        self.fc2 = (512, nClasses)

        # TODO: Implement module constructor.
        # Define network architecture as needed
        # Input imags will be 3 channels 256x256 pixels.
        # Output must be a nClasses Tensor.

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(self.flatten(x))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

        # TODO: 
        # Implement forward pass
        #  x is a BATCH_SIZEx3x256x256 Tensor
        #  return value must be a BATCH_SIZExN_CLASSES Tensor
