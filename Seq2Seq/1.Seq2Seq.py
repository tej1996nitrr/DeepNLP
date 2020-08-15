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
cuda = torch.device('cuda') 

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
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
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

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size =  source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device=cuda)
        hidden, cell = self.encoder(source)
        #grabbing start token
        x = target[0] 

        #sending to decoder word by word
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            #(N, eng_vocab_size)=output size
            best_guess = output.argmax(1)
            x  = target[t] if random.random()<teacher_force_ratio else best_guess

        return outputs

#%%
#TRAIING

#training hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

#model hyperparameters
load_model = False
device = torch.device('cuda' if torch.cuda.is_available())
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5

#tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data,valid_data,test_data),
                                                                        batch_size=batch_size,
                                                                        sort_within_batch=True,
                                                                        sort_key= lambda x:len(x.src),
                                                                        device=device
)
encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, encoder_dropout).to(device
)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, decoder_dropout).to(device
)
model = Seq2Seq(encoder, decoder).to(device)



