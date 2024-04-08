import os
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

START_TOKEN = "<SOS>"
END_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
DEFAULT_ROOT_DIR = "../dataset/Flickr8k/images/"
DEFAULT_ANNOTATIONS_DIR = "../dataset/Flickr8k/captions.txt"


spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {
            0: PAD_TOKEN, 1: START_TOKEN, 2: END_TOKEN, 3: UNKNOWN_TOKEN
        }
        self.stoi = {
            PAD_TOKEN: 0, START_TOKEN: 1, END_TOKEN: 2, UNKNOWN_TOKEN: 3
        }
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(test):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(test)]

    def build_vocabulary(self, sentences):
        frequencies = {}
        idx = 4
        for sentence in sentences:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] > self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi[token] if token in self.stoi else self.stoi[UNKNOWN_TOKEN]
                for token in tokenized_text]


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
        self.vocab.build_vocabulary(self.captions.tolist())

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
        num_caption.append(self.vocab.stoi[END_TOKEN])
        return image, torch.tensor(num_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        return images, targets


def get_loader(
        root_dir=DEFAULT_ROOT_DIR,
        annotation_file=DEFAULT_ANNOTATIONS_DIR,
        transform=None,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
):
    dataset = Flickr8k(root_dir, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi[PAD_TOKEN]
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx)
    )

    return loader, dataset


def test_dataloader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataloader, _ = get_loader(
        transform=transform,
    )

    for idx, (images, captions) in enumerate(dataloader):
        print(images.shape)
        print(captions.shape)


def test_captions_file(file="../dataset/Flickr8k/captions.txt"):
    df = pd.read_csv(file)
    print(df['caption'].head())
    print(df['caption'][1])
    print(df['caption'].iloc[1])


if __name__ == "__main__":
    # test_captions_file
    test_dataloader()
    print("🇨🇦Flickr Dataset Done")