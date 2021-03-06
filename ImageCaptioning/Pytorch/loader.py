#%%
import os
import pandas as pd
import spacy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import en_core_web_sm
# We want  to convert text -> numerical values
# 1. We need to setup a pytorch datset to load the data
# 2. We need a vocabulary mapping each word to index
# 3. Setup padding of every_batch (all examples  should be of same seq_length)
spacy_eng = en_core_web_sm.load()


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize_eng(text):
        return [tok.text.lower for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenize_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class FlickerDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # get img, caption columns
        self.imgs = self.df['image']
        self.captions = self.df['caption']

        # initialize vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        return img, torch.tensor(numericalized_caption)


class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueesze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        return imgs, targets


def get_loader(root_folder, annotation_file, transform, batch_size=32, num_workers=8, shuffle=True, pin_memory=True):
    dataset = FlickerDataset(root_folder, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                        pin_memory=pin_memory, collate_fn=Collate(pad_idx=pad_idx))
    return loader


# data_loader = get_loader(r'F:\VSCode\DeepNLP\ImageCaptioning\flicker8k',
#                          annotation_file=r'F:\VSCode\DeepNLP\ImageCaptioning\flicker8k\captions.txt', transform=None,
#                          shuffle=True)

def main():
    transform = transforms.Compose(
        [
        transforms.Resize((224,224)),
        transforms.ToTensor()
        ]
    )
    loader, dataset= get_loader(r'F:\VSCode\DeepNLP\ImageCaptioning\flickr8k',
                             annotation_file=r'F:\VSCode\DeepNLP\ImageCaptioning\flickr8k\captions.txt',
                             transform=transform,
                             shuffle=True)

    for idx, (imgs, captions) in enumerate(loader):
        print(imgs.shape)
        print(captions.shape)


if __name__ == "__main__":
    main()


# %%
'''consider only those words which occur at least 10 times. This helps the model become more robust to outliers and make less mistakes.'''
all_train_captions = []
for key, val in descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
word_count_threshold = 10
word_counts = {}
nsents=0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
        
vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d ' % len(vocab))