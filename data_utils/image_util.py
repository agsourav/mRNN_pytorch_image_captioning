import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

class BatchedImages(object):
    def __init__(self,  captions, image_shape, image_dir, batch_size): #image_shape (w,h)
        self.captions = captions
        self.batch_size = batch_size
        self.image_dir = image_dir
        self.shape = image_shape
    def dataLoader(self, train = True):
        transformer = transforms.Compose([
            transforms.Resize(self.shape),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.495, 0.466),
                                (0.225, 0.268, 0.225))
        ])
        batched_inp = self.captions.sample(self.batch_size)
        images = list(map(lambda x: x.split('#')[0], batched_inp.index))
        batched_images = []
        for image in images:
            img_path = os.path.join(self.image_dir, image)
            img = Image.open(img_path)
            img_tensor = transformer(img)
            batched_images.append(img_tensor.unsqueeze(0))
        batched_images = torch.cat(batched_images, dim = 0)
        batched_captions = batched_inp.values
        #dataset = torchvision.datasets.ImageFolder(self.image_dir, transform = transformer)
        #dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        return batched_images, batched_captions

    