import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml

# Define the Convolutional Neural Network architecture

# Load the configuration file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set up the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation for the input data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load the CIFAR-10 dataset and apply transformations
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['training']['batch_size'],
                                          shuffle=config['data']['shuffle'], num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=config['training']['batch_size'], shuffle=False,
                                         num_workers=2)

# Instantiate the CNN model
model = models.resnet101(pretrained=config['model']['pretrained']).to(device)
model.fc = nn.Linear(model.fc.in_features, config['model']['num_classes'])

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config['training']['learning_rate'], momentum=0.9)

# Training loop
for epoch in range(config['training']['epochs']):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs and labels
        inputs, labels = data[0].to(device), data[1].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # Print every 200 mini-batches
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print("Training finished.")

# Evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the test set: {(100 * correct / total):.2f}%')
