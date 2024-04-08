import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torch.nn as nn

# Methods for dealing with imbalanced datasets:
# 1. Oversampling
# 2. Class weighting

# Class weighting
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1, 15]))

# Oversampling
def get_loader(root_dir, batch_size):
    my_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root_dir, transform=my_transform)
    class_weights = []
    for root, subdir, files in os.walk(root_dir):
        if len(files) > 0:
            class_weights.append(1/len(files))
    sample_weights = [0] * len(dataset)

    for idx, (date, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader

def main():
    loader = get_loader(root_dir='../dataset/cat_and_dog', batch_size=8)

    for data, lables in loader:
        print(lables)

if __name__ == "__main__":
    main()
    print("🇨🇦Handle Imbalanced Dataset Done")