#%%
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
german = Field(tokenize=tokenizer_ger, lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokenizer_eng, lower=True, init_token='<sos>', eos_token='<eos')
train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)


# %%
