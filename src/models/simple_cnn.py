
import torch.nn as nn


class CIFAR10CNN(nn.Module):
    """
    CNN for CIFAR-10 with same architecture as "Dynamic Scheduling for Over-the-Air Federated Edge Learning with Energy Constraints, 2021"
    Total parameters: 258,898
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)   # 32×30×30
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)  # 32×28×28
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)  # 64×14×14 after pool
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)  # 64×12×12
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (batch, 3, 32, 32)
        x = self.relu(self.conv1(x))   # → (batch,32,30,30)
        x = self.relu(self.conv2(x))   # → (batch,32,28,28)
        x = self.pool(x)               # → (batch,32,14,14)
        x = self.relu(self.conv3(x))   # → (batch,64,12,12)
        x = self.relu(self.conv4(x))   # → (batch,64,10,10)
        x = self.pool(x)               # → (batch,64, 5,  5)
        x = x.view(x.size(0), -1)      # → (batch, 64*5*5)
        x = self.relu(self.fc1(x))     # → (batch,120)
        x = self.fc2(x)  # → (batch,10)
        return x
    