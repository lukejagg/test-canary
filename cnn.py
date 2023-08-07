import torch
import torch.nn as nn

# Define the Convolutional Neural Network architecture
class CNN(nn.Module):
    def __init__(self, num_classes):
        assert isinstance(num_classes, int) and num_classes > 0, "num_classes must be a positive integer"
        super(CNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # First ReLU activation function
        self.relu1 = nn.ReLU()
        # First pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Second ReLU activation function
        self.relu2 = nn.ReLU()
        # Fully connected layer
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x