import torch.nn as nn

# Define the Convolutional Neural Network architecture
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        assert isinstance(num_classes, int) and num_classes > 0, "num_classes must be a positive integer"
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        # Calculate the size of the output feature map
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        x = x.view(x.size(0), -1)
        self.fc = nn.Linear(num_features, self.num_classes)
        x = self.fc(x)
        return x