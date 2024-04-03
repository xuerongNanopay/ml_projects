import torch
import torch.utils as save_image
from torchvision import datasets, transforms

my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[], std=[]) # Find mean and std first
])
mnist_dataset = datasets.MNIST(root='dataset/', train=True, transform=my_transforms, download=True)

# img_num = 0
# for img, label in mnist_dataset:


if __name__ == "__main__":
    print("🇨🇦Augmentation Done")