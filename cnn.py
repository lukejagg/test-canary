import torch.nn as nn

# Define the Convolutional Neural Network architecture
class CNN(nn.Module):
    def __init__(self, num_classes, activation_function=None):
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("`num_classes` must be a positive integer.")
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.activation_function = activation_function
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        if x.size(1) * x.size(2) * x.size(3) != 32 * 8 * 8:
            raise ValueError("`x` must be able to be reshaped to (32 * 8 * 8).")
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.activation_function is not None:
            x = self.activation_function(x)
        return x