import os
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as Dataset
from PIL import Image

START_TOKEN = "<SOS>"
END_TOKEN = "<EOS>"


class Flickr8k(Dataset):
    """
    Download: https://www.kaggle.com/datasets/adityajn105/flickr8k/data
    """
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # Get image, caption columns
        self.images = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolst())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        image_id = self.images[index]
        image = Image.open(os.path.join(self.root_dir, image_id)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        num_caption = [self.vocab.stoi[START_TOKEN]]
        num_caption += self.vocab.numericalize(caption)
        num_caption += self.vocab.stoi[END_TOKEN]
        return image, torch.tensor(num_caption)


def test_captions_file(file="../dataset/Flickr8k/captions.txt"):
    df = pd.read_csv(file)
    print(df['caption'].head())
    print(df['caption'][1])
    print(df['caption'].iloc[1])


if __name__ == "__main__":
    test_captions_file()