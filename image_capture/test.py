"""
Model Test
"""
import torch
from model import CNNtoRNN
import torchvision.transforms as transforms
from datasets.Flickr import get_loader
from PIL import Image

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)


def load_image(image_dir):
    image = Image.open(image_dir)
    return TRANSFORM(image)


def load_model(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")

    _, dataset = get_loader()
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    vocab_size = len(dataset.vocab)
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model, dataset


def test(image_dir):
    image = load_image(image_dir).unsqueeze(0)
    print(image.shape)
    model, dataset = load_model("my_checkpoint.pth.tar")
    print(model.caption_image(image, dataset.vocab))


if __name__ == "__main__":
    # test("../dataset/playground_images/cat.jpg")
    test("../dataset/Flickr8k/images/10815824_2997e03d76.jpg")
    print("🇨🇦Done")
