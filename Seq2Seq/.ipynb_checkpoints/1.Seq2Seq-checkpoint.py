# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
import de_core_news_sm
import en_core_web_sm
# from torch.utils.tensorboard import SummaryWriter

# %%
spacy_eng = en_core_web_sm.load()
spacy_ger = de_core_news_sm.load()

# %%
'''Tokenizing Text'''


def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


# %%
german = Field(tokenize=tokenizer_ger, lower=True,
               init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokenizer_eng, lower=True,
                init_token='<sos>', eos_token='<eos')
train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)
german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=1000, min_freq=2)

# %%


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn, nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, p)

    def forward(self, x):
        # x_shape:(seq_length, N) N=batch size
        embedding = self.dropout(self.embedding(x))
        #embedding_shape: (seq_length, N, embedding_size)
        outputs,(hidden, cell) = self.rnn(embedding)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn, nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, p)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x,hidden, cell):
        #shape of x :(N) but we want (1,N)
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        #embedding shape : ( 1, N, embedding_size)
        outputs,(hidden,cell) = self.rnn(embedding, (hidden, cell))
        #shape of outputs: (1,N, hidden_size)
        predictions = self.fc(outputs)
        #shape of predictions: (1,N, length_of_vocab)
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell

# class Seq2Seq(nn.Module):
#     def __init__(self, encoder, decoder):
#         super(Seq2Seq, self).__init__()
#         self.decoder = decoder
#         self.encoder = encoder

#     def forward(self, source, target, teacher_force_ratio=0.5):

