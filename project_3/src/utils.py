import os
from numpy import double
import pandas as pd
import torch
from torch import nn
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


def plot_images(images, titles, model_name):
	axes = []
	plt.figure(figsize=(10, 16))
	for i in range(len(images)):
		axes.append(plt.subplot(1, len(images), i+1))
		axes[i].imshow(images[i], cmap="gray")
		axes[i].set_title(titles[i], fontsize=15)
	plt.savefig(f"../out/{model_name}.png")
	plt.close()


class NoiseDataset(Dataset):
    def __init__(
        self,
        csv_file="TrainingDataSet.csv",
        root_dir_noisy="TrainingDataSet",
        root_dir_ref="./",
        transform=None,
    ):
        # read csv file
        self.name_csv = pd.read_csv(csv_file)

        # store attributes
        self.root_dir_noisy = root_dir_noisy
        self.root_dir_ref = root_dir_ref
        self.transform = transform

    def __len__(self):
        return len(self.name_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image filenames
        ref_img_name = os.path.join(self.root_dir_ref, self.name_csv.iloc[idx, 0])
        noisy_img_name = os.path.join(self.root_dir_noisy, self.name_csv.iloc[idx, 2])

        # load images
        ref_image = torch.from_numpy(imread(ref_img_name)[np.newaxis,:,:].astype(double)).to(device) / 255.0
        noisy_image = torch.from_numpy(imread(noisy_img_name)[np.newaxis,:,:].astype(double)).to(device) / 255.0
        # apply transforms
        if self.transform:
            ref_image = self.transform(ref_image)
            noisy_image = self.transform(noisy_image)
        return noisy_image, ref_image

class CNN_1Layer(nn.Module):
    def __init__(self):
        super(CNN_1Layer, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding="same"),
        )

    def forward(self, x):
        x = self.convolution(x)
        return x


class CNN_5Layer(nn.Module):
    def __init__(self):
        super(CNN_5Layer, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding="same"),
            torch.nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"),
            torch.nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"),
            torch.nn.BatchNorm2d(num_features=128),
			nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding="same"),
            torch.nn.BatchNorm2d(num_features=64),
			nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding="same"),
            torch.nn.BatchNorm2d(num_features=1),
        )

    def forward(self, x):
        x = self.convolution(x)
        return x

class CNN_5Layer_NL(nn.Module):
    def __init__(self):
        super(CNN_5Layer_NL, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding="same"),
			nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"),
			nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"),
			nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=128),
			nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding="same"),
			nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=64),
			nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding="same"),
			nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=1),
        )

    def forward(self, x):
        x = self.convolution(x)
        return x

