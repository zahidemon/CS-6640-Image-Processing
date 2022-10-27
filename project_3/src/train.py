import os
import argparse

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from utils import NoiseDataset, CNN
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_loop(dataloader, model, loss_fn, optimizer):
    """
    trains your model for an epoch
    returns an array of loss values over the training epoch
    """
    loss_array = []
    accuracies = []
    model.train()

    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()

        if batch % 200 == 0:  # print some status info
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        accuracies.append((pred.argmax(1) == y).type(torch.float).sum().item())

        loss_array.append(loss.item())
    return np.mean(loss_array), np.mean(accuracies)


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

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

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
    BATCH_SIZE = 16
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

    # TODO: define your models
    model = CNN()

    # TODO: define your optimizer using learning rate and other hyperparameters
    learning_rate = 1e-3
    epochs = 5

    # initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()  # use MSE loss for this project

    train_losses, test_losses, train_acc, test_acc = [], [], [], []

    if not FLAGS.demo:
        for t in range(EPOCHS):
            print(f"Epoch {t+1}\n-------------------------------")
            tr_loss, tr_acc,  = train_loop(train_dataloader, model, loss_fn, optimizer)
            ts_loss, ts_acc = test_loop(test_dataloader, model, loss_fn)
            train_losses.append(tr_loss)
            train_acc.append(tr_acc)
            test_losses.append(ts_loss)
            test_acc.append(ts_acc)
        print("Done!")

    # TODO: save model
    torch.save(model, '../models/model.pth')
    # TODO: plot line charts of training and testing metrics (loss, accuracy)
    epochs = range(1, EPOCHS)

    # Plot and label the training and validation loss values
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, test_acc, label='Test Accuracy')

    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Set the tick locations
    plt.xticks(list(range(0, EPOCHS, 2)))

    # Display the plot
    plt.legend(loc='best')
    plt.savefig("../out/plot.png")
    plt.show()

    # TODO: plot some of the testing images by passing them through the trained model

    if FLAGS.demo:
        pass
        # TODO: set up a demo with a small subset of images
