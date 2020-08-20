import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stemming(word):
    return stemmer.stem(word.lower())


if __name__ == "__main__":
    tokenized = tokenize("HMmm how you doing?")
    print(tokenized)
    stemmed = []
    for w in tokenized:
        stemmed.append(stemming(w))
    print(stemmed)
