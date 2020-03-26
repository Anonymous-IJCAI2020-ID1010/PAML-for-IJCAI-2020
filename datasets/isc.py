"""
An example for dataset loaders, starting with data loading including all the functions that either preprocess or postprocess data.
"""
import imageio
import torch
import torchvision.utils as v_utils
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder 

class Isc:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        if config.data_mode == "imgs":

            # train_datasets = ImageFolder(self.config.train_folder, transform=make_raw_transform(
            #     sz_resize = self.config.sz_resize, sz_crop = self.config.sz_crop, mean = [0.8324, 0.8109, 0.8041], std = [0.2206, 0.2378, 0.2444]))
            # query_datasets = ImageFolder(self.config.query_folder, transform=make_raw_transform(
            #     sz_resize = self.config.sz_resize, sz_crop = self.config.sz_crop, mean = [0.8324, 0.8109, 0.8041], std = [0.2206, 0.2378, 0.2444], is_train=False))
            # gallery_datasets = ImageFolder(self.config.gallery_folder, transform=make_raw_transform(
            #     sz_resize = self.config.sz_resize, sz_crop = self.config.sz_crop, mean = [0.8324, 0.8109, 0.8041], std = [0.2206, 0.2378, 0.2444], is_train=False))
            
            self.train_transforms = transforms.Compose([
                
                transforms.Resize(int(self.config.sz_resize*1.1)),
                transforms.RandomRotation(10),
                transforms.RandomCrop(self.config.sz_crop),
                transforms.RandomHorizontalFlip(),
                # transforms.Pad(cfg.INPUT.PADDING),
                transforms.RandomCrop(self.config.sz_crop),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.8324, 0.8109, 0.8041], std = [0.2206, 0.2378, 0.2444]),
                RandomErasing()

                # ScaleIntensities(*None) if None is not None else Identity(),
                # transforms.Lambda(lambda x: x[[2, 1, 0], ...]) if False else Identity()
                ])

            

            self.test_transforms = transforms.Compose([
                transforms.Resize(int(self.config.sz_resize*1.1)),
                transforms.CenterCrop(self.config.sz_crop),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.8324, 0.8109, 0.8041], std = [0.2206, 0.2378, 0.2444])
                ])

            train_datasets = ImageFolder(self.config.train_folder, transform=self.train_transforms)   
            query_datasets = ImageFolder(self.config.query_folder, transform=self.test_transforms) 
            gallery_datasets = ImageFolder(self.config.gallery_folder, transform=self.test_transforms) 

            if self.config.use_sampler:
                raise NotImplementedError("This use_sampler is not implemented YET")
            else:
                self.train_loader = DataLoader(train_datasets,
                    batch_size = self.config.batch_size,
                    shuffle = True,
                    drop_last = True,
                    num_workers = self.config.num_workers,
                    pin_memory = self.config.pin_memory,
                    )
            self.query_loader = DataLoader(query_datasets,
                batch_size = self.config.test_batch_size,
                shuffle = False,
                num_workers = self.config.num_workers,
                pin_memory = self.config.pin_memory,
                )
            self.gallery_loader = DataLoader(gallery_datasets,
                batch_size = self.config.test_batch_size,
                shuffle = False,
                num_workers = self.config.num_workers,
                pin_memory = self.config.pin_memory,
                )

        elif config.data_mode == "numpy":
            raise NotImplementedError("This mode is not implemented YET")

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def plot_samples_per_epoch(self, batch, epoch):
        """
        Plotting the batch images
        :param batch: Tensor of shape (B,C,H,W)
        :param epoch: the number of current epoch
        :return: img_epoch: which will contain the image of this epoch
        """
        img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
        v_utils.save_image(batch,
                           img_epoch,
                           nrow=4,
                           padding=2,
                           normalize=True)
        return imageio.imread(img_epoch)

    def make_gif(self, epochs):
        """
        Make a gif from a multiple images of epochs
        :param epochs: num_epochs till now
        :return:
        """
        gen_image_plots = []
        for epoch in range(epochs + 1):
            img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
            try:
                gen_image_plots.append(imageio.imread(img_epoch))
            except OSError as e:
                pass

        imageio.mimsave(self.config.out_dir + 'animation_epochs_{:d}.gif'.format(epochs), gen_image_plots, fps=2)

    def finalize(self):
        pass

def make_raw_transform(sz_resize = 256, sz_crop = 227, mean = [0.4707, 0.4601, 0.4549], 
        std = [0.2767, 0.2760, 0.2850], rgb_to_bgr = False, is_train = True, 
        intensity_scale = None):
    return transforms.Compose([
        transforms.Compose([ # train: horizontal flip and random resized crop
            transforms.Resize(int(sz_resize*1.1)),
            transforms.RandomRotation(10),
            transforms.RandomCrop(sz_crop),
            #transforms.RandomResizedCrop(sz_crop),
            transforms.RandomHorizontalFlip(),
        ]) if is_train else transforms.Compose([ # test: else center crop
        ]),
        transforms.ToTensor(),
        ScaleIntensities(
            *intensity_scale) if intensity_scale is not None else Identity(),
        transforms.Normalize(
            mean=mean,
            std=std,
        ),
        transforms.Lambda(
            lambda x: x[[2, 1, 0], ...]
        ) if rgb_to_bgr else Identity()])

class ScaleIntensities():
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __call__(self, tensor):
        tensor = (
            tensor - self.in_range[0]
        ) / (
            self.in_range[1] - self.in_range[0]
        ) * (
            self.out_range[1] - self.out_range[0]
        ) + self.out_range[0]
        return tensor

class Identity(): # used for skipping transforms
    def __call__(self, im):
        return im

import math
import random


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.8324, 0.8109, 0.8041)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img