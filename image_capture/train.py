import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_example
from datasets.Flickr import get_loader
from model import CNNtoRNN

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    train_loader, dataset = get_loader(transform=transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = True
    save_model = True

    # Hyperparameters.
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 1

    # tensorboard
    # writer = SummaryWriter("runs/flickr")
    step = 0

    # Initialize model
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("=== Start training...")
    print(f"=== Vocabulary size: {vocab_size}")

    if load_model:
        step = load_checkpoint("my_checkpoint.pth.tar", model, optimizer)
        print(f"step: {step}")
    model.train()

    for epoch in range(num_epochs):

        # captions is Sequence First
        for idx, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)
            # print('vvv', playground_images)
            # why -1? First Sequence will be input from Encoder
            outputs = model(images, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            # writer.add_scalar("loss", loss.item(), global_step=step)

            print(f"loss: {loss.item():.4f}")
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }

            save_checkpoint(checkpoint, "my_checkpoint.pth.tar")


if __name__ == "__main__":
    train()
    print("🇨🇦Done")
