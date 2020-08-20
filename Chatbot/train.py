import json
from nlp_utils import stemming, tokenize, bag_of_words
import numpy as np

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', ',', '!', '.']
all_words = [stemming(w) for w in all_words if w not in ignore_words]
print(all_words)
tags = sorted(set(tags))
print(tags)

x_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    label = tags.index(tag)

    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

if __name__ == "__main__":
    pass
