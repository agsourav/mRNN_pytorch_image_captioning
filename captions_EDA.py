from data_utils.caption_util import PreprocessCaptions
import pandas as pd

captions_file = 'data/captionSeries.pkl'
captions_df = 'data/caption_df.pkl'

preprocess = PreprocessCaptions()
captions = preprocess.load_captions(captions_file)
captions_length = preprocess.load_caption_df(captions_df)

print(captions_length.describe())
