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

class RNN(nn.Module):
    """
        RNN model for MNIST
    """