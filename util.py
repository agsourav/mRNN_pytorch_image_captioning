import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Multi modal rnn architecture')
    parser.add_argument('--img-dir', dest = 'image_dir', help = 'image directory path',
    default = 'data/Flicker8k_Dataset', type = str)
    parser.add_argument('--bs', dest = 'batch_size', help = 'batch size',
    default = 1)

    return parser.parse_args()