import torch
import torch.nn as nn
import torch.optim as optim  # Adam, SGD
import torch.nn.functional as F  # Active Function, Relu
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 28 * 28
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 20

# Create Fully Connected Network
class NN(nn.Module):
    """
        Fully Connect model for MNIST
    """
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on testing data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            x = x.reshape(x.shape[0], -1)
            y = y.to(device=device)

            scores = model(x)  # 64 * 10
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}%')

    model.train()

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = NN(input_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Get to correct shape.
        data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        # Set gradient to zero for each batch.
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)

# Check accuracy on training & test to see how good our model.

if __name__ == "__main__":
    print("🇨🇦Fullynet Done")
