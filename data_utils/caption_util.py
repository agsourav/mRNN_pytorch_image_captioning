import torch
from gensim import corpora
import nltk.tokenize as T

class Vocabulary(object):
    def __init__(self, text_corpus):

        self.text_corpus = text_corpus
        self.dictionary = None

    def generate_dictionary(self):
        extra_tokens = [['<pad>','<start>','<stop>']]
        d1 = corpora.Dictionary(extra_tokens)
        tokens = [T.word_tokenize(sentence) for sentence in T.sent_tokenize(self.text_corpus)]
        self.dictionary = corpora.Dictionary(tokens)
        self.dictionary.merge_with(d1)

    def word2idx(self, word):
        return self.dictionary.token2id[word]

    def idx2word(self, idx):
        return self.dictionary[idx]

    


    
