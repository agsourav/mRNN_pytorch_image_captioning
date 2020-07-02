import torch
from gensim import corpora
import nltk.tokenize as T
import time
import pandas as pd
import numpy as np
class Vocabulary(object):
    def __init__(self):
        self.dictionary = None

    def generate_dictionary(self, text_corpus):
        extra_tokens = [['<pad>','<start>','<stop>']]
        d1 = corpora.Dictionary(extra_tokens)
        tokens = [T.word_tokenize(sentence) for sentence in T.sent_tokenize(text_corpus)]
        self.dictionary = corpora.Dictionary(tokens)
        self.dictionary.merge_with(d1)

    def word2idx(self, word):
        return self.dictionary.token2id[word]

    def idx2word(self, idx):
        return self.dictionary[idx]


class PreprocessCaptions(object):
    def __init__(self):
        super(PreprocessCaptions, self).__init__()

    def load_init(self, file_path, captionSeries_file):
        start = time.time()
        with open(file_path, 'r') as f:
            data = f.read()
        self.generate_series(data, captionSeries_file)
        end = time.time()
        print('time taken to load a file: {0:2.5f}'.format(end - start))
        print('file saved!!')
    
    def generate_series(self, data, captionSeries_file):
        data_lines = data.split('\n')
        captions = list(map(lambda x: x.split('\t'), data_lines))
        image_file, captions = zip(*captions)
        captions = pd.Series(captions, index = image_file)
        captions.to_pickle(captionSeries_file)

    def load_captions(self, file_path):
        captions = pd.read_pickle(file_path)
        return captions

    def generate_lengths(self, captions, caption_df_file):
        caption_length = captions.copy()
        caption_word_counts = captions.copy()
        start = time.time()
        for i, caption in enumerate(captions):
            caption_length[i] = len(caption)
            caption_word_counts[i] = len(caption.split(' '))
        caption_df = {'length': caption_length.astype(dtype = np.int32), 'word_count': caption_word_counts.astype(dtype = np.int32)}
        caption_df = pd.DataFrame(caption_df)
        end = time.time()
        print('time to generate caption lengths: {0:2.5f}'.format(end - start))
        caption_df.to_pickle(caption_df_file)

    def load_caption_df(self, caption_df):
        caption_lengths = pd.read_pickle(caption_df)
        return caption_lengths


