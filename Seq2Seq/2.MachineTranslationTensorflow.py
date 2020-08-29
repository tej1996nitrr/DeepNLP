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
