import torch.nn as nn

# Define the Convolutional Neural Network architecture
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # Third convolutional layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        # Fully connected layer
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        # Check the shape of the input tensor
        if x.shape[1:] != (3, 32, 32):
            raise ValueError("Input tensor has incorrect shape!")
        # First convolutional layer
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        # Second convolutional layer
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        # Third convolutional layer
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool(x)
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # Fully connected layer
        x = self.fc(x)
        # Activation function
        x = nn.functional.softmax(x, dim=1)
        return x