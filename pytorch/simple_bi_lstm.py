import torch
import torch.nn as nn
import torch.optim as optim  # Adam, SGD
import torch.nn.functional as F  # Active Function, Relu
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE: ", device)

if __name__ == "__main__":
    print("🇨🇦BiLSTM Done")