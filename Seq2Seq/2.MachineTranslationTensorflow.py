#%%
'''Downloading Dataset'''
import os
import sys
import utils
import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
from scipy.spatial.distance import cdist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# utils.imdbdataset.maybe_download_and_extract()
# %%
x_train_text, y_train = utils.imdbdataset.load_data(train=True)
x_test_text, y_test = utils.imdbdataset.load_data(train=False)
# %%
print("Train-set Size: ", len(x_train_text))
print("Test-set Size: ", len(x_test_text))
data_set = x_test_text + x_train_text
# %%
print(x_train_text[1])
print(y_train[1])
y_train = np.array(y_train)
y_test = np.array(y_test)
# %%
'''Tokenizer'''
# tokenizer to only use the 10000 most popular words from the data-set.
num_words = 10000
tokenizer = Tokenizer(num_words=num_words)  
tokenizer.fit_on_texts(data_set)
# %%
print(len(tokenizer.word_index))
# %%
x_train_tokens = tokenizer.texts_to_sequences(x_train_text)
# %%
print(x_train_text[1])
# %%
print(np.array(x_train_tokens[1]))
# %%
x_test_tokens = tokenizer.texts_to_sequences(x_test_text)
# %%
'''Padding and Truncating data'''
# we here make a compromise  and use sequence-length that covers most of the data
# and we will then truncate longer sequences and pad shorter sequences.
num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)
# average number of tokens in a sequence 
print("average number of tokens in a sequence ", np.mean(num_tokens))
# %%
# max  number of tokens
print(np.max(num_tokens))

# %%
# the max number of tokens we will allow is set to the average plus 2 standard deviations.
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
print(max_tokens)

# %%
# This covers about 95% of the data-set.
np.sum(num_tokens < max_tokens) / len(num_tokens)
# %%
# padding or truncating the sequences
pad = 'pre' # adding zeros first, if pad='post, zeros are added at the end
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
                            padding=pad, truncating=pad)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)
print(x_train_pad.shape)
print(x_test_pad.shape)
print("Example", np.array(x_train_tokens[1]), x_train_pad[1])
# %%
'''Inverse Mapping'''
idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))
def tokens_to_string(tokens):
    # Map from tokens back to words.
    words = [inverse_map[token] for token in tokens if token != 0]
    
    # Concatenate all words.
    text = " ".join(words)

    return text
    
print("Example for reverse mapping", x_train_text[1],".......recreated this text......",
       tokens_to_string(x_train_tokens[1]))
# %%
