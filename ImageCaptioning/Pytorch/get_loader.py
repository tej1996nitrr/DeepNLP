import os
import pandas as pd
import spacy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# We want  to convert text -> numerical values
# 1. We need to setup a pytorch datset to load the data
# 2. We need a vocabulary mapping each word to index
# 3. Setup padding of every_batch (all examples  should be of same seq_length)
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0,"<SOS>":1,"<EOS>":2 ,"<UNK>":3}
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
                    frequenies[word]=1
                else:
                    frequencies[word]+=1
                if frequencies[word]== self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx]=word
                    idx+=1
    
    def numercalize(self, text):
        tokenized_text =  self.tokenize_eng(text)
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
        self.captions = self.df['captions']

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
        numericalized_caption+=self.vocab.numericalize(caption)
        numericalized_caption.append(self.voab.stoi["<EOS>"])
        return img,torch.tensor(numericalized_caption)
        


