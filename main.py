from data_utils.caption_util import *
from data_utils.image_util import *
from mrnn import Model, Evaluate
from gensim.corpora import Dictionary
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

captionSeries_file = 'data/captionSeries.pkl'       #loading pickle file formats are significantly faster
caption_file = 'data/Flickr8k.lemma.token.txt'
caption_df_file = 'data/caption_df.pkl'
dictionary_file = 'data/captions/dictionary.txt'
image_dir = 'data/flickr_dataset/Flicker8k_Dataset'

#preprocess.load_init(caption_file, captionSeries_file)
#caption_df = preprocess.load_caption_df(caption_df_file)

dictionary = Dictionary.load_from_text(dictionary_file)
vocab = Vocabulary()
vocab.dictionary = dictionary
vocab_size = len(vocab.dictionary)

batch_size = 1
image_shape = (128,128)
start = time.time()     #start
Images = BatchedImages(vocab, captionSeries_file, image_shape, image_dir = image_dir, batch_size = batch_size)
images, captions_ = Images.dataLoader()
end = time.time()       #end
print('batch loaded in time : ',images.shape, (end - start))

inp_word = torch.tensor([vocab.word2idx('<start>')] * batch_size)
inp_word = torch.reshape(inp_word, (batch_size, -1))
print(inp_word)

mrnn = Model(input_dim = 128, embed_dim = 128, hidden_dim = 128, output_dim = 20, feature_dim = 128, vocab_size = vocab_size)
max_caption_len = 35
stop_val = torch.tensor(vocab.word2idx('<stop>'))
stop_val = torch.reshape(stop_val, (1,1))
model_out, caption_out = mrnn(inp_word, images, max_caption_len)
output = torch.argmax(caption_out, dim = 2, keepdim = True)

caption_gen = [list(map(lambda x: vocab.idx2word(x.item()), output[:,i,0])) for i in range(output.shape[1])]
true_caption = [list(map(lambda x: vocab.idx2word(x), captions_[i])) for i in range(len(captions_))]
print('before training')
#print(caption_gen)
print('-'*20)
'''
plt.figure()
plt.imshow(images[0].permute(1,2,0))
#print('Generated caption: ',output, output.shape, caption_out.shape)
'''
optim = torch.optim.SGD(mrnn.parameters(), lr = 0.01)
criterion = nn.NLLLoss()
eval = Evaluate(mrnn, Images, criterion, optim, 50)
eval(inp_word)

model_out, caption_out = mrnn(inp_word, images, max_caption_len)
output = torch.argmax(caption_out, dim = 2, keepdim = True)

caption_gen = [list(map(lambda x: vocab.idx2word(x.item()), output[:,i,0])) for i in range(output.shape[1])]
true_caption = [list(map(lambda x: vocab.idx2word(x), captions_[i])) for i in range(len(captions_))]
print('after training')
print('Generated caption: ',caption_gen)
print('True caption: ', true_caption)
print('-'*20)