from data_utils.caption_util import *
from mrnn import Model
import torch.nn as nn

'''
text_corpus = 'Beautiful girl is dancing in the park. Isn\'t she beautiful? She is enjoying herself.'
vocab = Vocabulary(text_corpus)
vocab.generate_dictionary()

print(vocab.dictionary)
print(vocab.word2idx('<pad>'))
'''
batch_size = 4
inp = torch.randint(low = 0, high = 255, size = (batch_size,1))  # [batch_size, word_int]
inp_img = torch.rand((batch_size, 3, 256, 256))
embed_word = nn.Embedding(256, 128)
word_inp = embed_word(inp)
mrnn = Model(128, 128, 128, 20, 128)
caption = mrnn(word_inp.permute(1,0,2), inp_img, 5)
print(caption.shape)

