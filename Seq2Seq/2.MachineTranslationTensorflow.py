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
x_train_text[1]
y_train[1]
# %%
