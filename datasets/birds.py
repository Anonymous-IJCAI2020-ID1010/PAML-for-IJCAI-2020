"""
An example for dataset loaders, starting with data loading including all the functions that either preprocess or postprocess data.
"""
import imageio
import torch
import torchvision.utils as v_utils
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder 

class Bird:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        if config.data_mode == "imgs":
            self.train_transforms = transforms.Compose([
                transforms.Resize(int(self.config.sz_resize*1.1)),
                transforms.RandomRotation(10),
                transforms.RandomCrop(self.config.sz_crop),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.4707, 0.4601, 0.4549], std = [0.2767, 0.2760, 0.2850])])

            self.test_transforms = transforms.Compose([
                transforms.Resize(int(self.config.sz_resize*1.1)),
                transforms.CenterCrop(self.config.sz_crop),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.4707, 0.4601, 0.4549], std = [0.2767, 0.2760, 0.2850])])

            train_datasets = ImageFolder(self.config.train_folder, transform=self.train_transforms)
            test_datasets = ImageFolder(self.config.test_folder, transform=self.test_transforms)

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
            self.valid_loader = DataLoader(train_datasets,
                batch_size = self.config.test_batch_size,
                shuffle = False,
                num_workers = self.config.num_workers,
                pin_memory = self.config.pin_memory,
                )
            self.test_loader = DataLoader(test_datasets,
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

