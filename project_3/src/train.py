import os
import argparse

import numpy as np
from utils import CNN_1Layer
import torch
from matplotlib import axis, pyplot as plt
from torch import nn
from random import randint

from utils import NoiseDataset, CNN_1Layer, CNN_5Layer, CNN_5Layer_NL, plot_images
from torch.utils.data import DataLoader, Subset

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Running on ', device)


def train_loop(dataloader, model, loss_fn, optimizer):
    """
    trains your model for an epoch
    returns an array of loss values over the training epoch
    """
    loss_array = []
    model.train()

    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        # print(X.shape, y.shape)
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        loss_array.append(loss.item())
        if batch % 50 == 0:  # print some status info
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return np.mean(loss_array)


def test_loop(dataloader, model, loss_fn):
    """
    tests your model on the test set
    returns average MSE and accuracy
    """
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            # move images to GPU if needed
            X, y = X.to(device), y.to(device)

            # compute prediction and loss
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \nAvg loss: {test_loss:>8f} \n")

    return test_loss, 100 * correct


if __name__ == "__main__":
    # parse --demo flag, if not there FLAGS.demo == False
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", dest="demo", action="store_true")
    parser.set_defaults(demo=False)
    FLAGS, unparsed = parser.parse_known_args()

    # make output directory
    if not os.path.exists("./out/"):
        os.makedirs("./out/")
    # make models directory
    if not os.path.exists("./models/"):
        os.makedirs("./models/")

    # tweak these constants as you see fit, or get them through 'argparse'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 4
    EPOCHS = 5
    DATASET_LOC = "../Data/nn_data/pokemon/"
    LEARNING_RATE = 1

    train_dataset = NoiseDataset(
        csv_file=DATASET_LOC + "training.csv",
        root_dir_noisy=DATASET_LOC + "training",
    )
    test_dataset = NoiseDataset(
        csv_file=DATASET_LOC + "testing.csv",
        root_dir_noisy=DATASET_LOC + "testing",
    )

    # define dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # define your models
    model = CNN_5Layer()
    model = model.to(float).to(device)
    print(model)

    # define your optimizer using learning rate and other hyperparameters
    learning_rate = 1e-3
    epochs = 5

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()  # use MSE loss for this project

    train_losses, test_losses = [], []

    if not FLAGS.demo:
        for t in range(EPOCHS):
            print(f"Epoch {t+1}\n-------------------------------")
            tr_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
            ts_loss = test_loop(test_dataloader, model, loss_fn)
            train_losses.append(tr_loss)
            test_losses.append(ts_loss)
        print("Done!")

    # save model
    torch.save(model, '../models/model_5Layer.pth')
    # plot line charts of training and testing metrics (loss, accuracy)
    epochs = range(1, EPOCHS+1)

    # Plot and label the training and validation loss values
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Test Loss')

    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Set the tick locations
    plt.xticks(list(range(0, EPOCHS, 2)))

    # Display the plot
    plt.legend(loc='best')
    plt.savefig("../out/plot_5Layer.png")
    plt.show()

    # TODO: plot some of the testing images by passing them through the trained model
    # model = torch.load("../models/model_5Layer.pth")
    # print(model)
    test_data_size = len(test_dataloader)
    random_indexes = [randint(1, test_data_size-1) for _ in range(5)]
    sample_set = Subset(test_dataset, random_indexes)
    
    sample_dataloader = DataLoader(sample_set, batch_size=1, shuffle=True)
    print(len(sample_dataloader))
    with torch.no_grad():
        for i, (X, y) in enumerate(sample_dataloader):
            # move images to GPU if needed
            X, y = X.to(device), y.to(device)
            # compute prediction and loss
            print(X.shape, y.shape)
            pred = model(X)
            plot_images(
                [
                    y.cpu().detach().numpy().reshape((475, 475)),
                    X.cpu().detach().numpy().reshape((475, 475)),
                    pred.cpu().detach().numpy().reshape((475, 475))
                ],
                ["Ground Truth", "Noisy", "Predicted"], f"model_5Layer_{i}"
                )

    if FLAGS.demo:
        pass
        # TODO: set up a demo with a small subset of images
