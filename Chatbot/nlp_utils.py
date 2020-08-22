import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stemming(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stemming(word) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1.0
    return bag


if __name__ == "__main__":
    tokenized = tokenize("HMmm how you doing?")
    stemmed = []
    for w in tokenized:
        stemmed.append(stemming(w))
    sentence = ["Hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thanks", "cool"]
    bag = bag_of_words(sentence, words)
    print(bag)
