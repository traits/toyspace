import torch
from torch import nn, optim
from torch.utils import data
from torch.utils.data import DataLoader
import torch.nn.functional as func
from models import *


def train(model, dataset, epochs, bs, lr, device):
    """
    Train single net on images/labels pair
    
    Parameters: 
        :model: used network model
        :dataset: used dataset
        :epochs: number of epochs
        :bs: batch size
        :lr: learning rate
        :device: utilized torch device
    """
    model = model.to(device)
    model.train()

    loader = DataLoader(dataset, batch_size=bs)

    criterion = nn.NLLLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    epochs = epochs
    for e in range(epochs):
        running_loss = 0
        for images, labels in loader:
            # reset gradients
            optimizer.zero_grad()

            images = model.adjustImages(images)
            images = images.to(device)
            # calls classes forward() (cf. next comment)
            output = model(images)
            # https://pytorch.org/docs/stable/nn.html#loss-functions
            # callable of Loss classes, ess. (not quite)
            # calls the classes forward() function
            loss = criterion(output, labels)

            # backpropagating
            loss.backward()

            # optimizes weights here
            optimizer.step()

            running_loss += loss.item()
        else:
            print(f"Epoch {e} - Training loss: {running_loss/len(loader)}")


def test(model, dataset, device):
    """
    Test single net on images and return predicted labels
    
    Parameters:
        :model: used network model
        :dataset: used dataset
        :device: utilized torch device
    Returns:    
        np.array(int8) of labels
    """

    model.eval()
    test_loss = 0
    correct = 0
    bs = 30
    loader = DataLoader(dataset, batch_size=bs)
    predicted_labels = np.zeros([len(dataset)], dtype=np.uint8)
    with torch.no_grad():
        i = 0
        for data in loader:
            data = data[0]
            # print(data.shape)
            data = model.adjustImages(data)
            data = data.to(device)
            output = model(data)

            # (10, bs) tensor: rows contain weighted probs
            # for every number
            output = func.softmax(output, 1).cpu()
            # array of row maxima in output
            maxp = torch.max(output, 1)[0].numpy()
            # indexes of these maxiama
            maxi = torch.max(output, 1)[1].numpy()

            predicted_labels[i * bs : i * bs + len(output)] = maxi
            i = i + 1
            # print(predicted_labels)
            # print(f"inferred {i*bs} from {len(images)}")
        print(f"predicted labels: {len(predicted_labels)}")
    return predicted_labels


def run(model, train_data, epochs, bs, lr, test_data, device):
    """
    Run complete training-test cycle for particular model
    
    Parameters:
        :model: used network model
        :train_data: used dataset
        :epochs: number of epochs
        :bs: batch size
        :lr: learning rate
        :train_data: used test dataset
        :device: utilized torch device

    """
    train(model, train_data, epochs, bs, lr, device)
    print("finished training")
    pred_labels = test(model, test_data, device)
    print("finished test")
    return pred_labels


def testEnsemble(ensemble, dataset, device):
    """
    Test ensemble of binary classificators

    Parameters:
        :ensemble: list of binary classificator models
        :dataset: used dataset
        :device: utilized torch device
    """
    if not isinstance(ensemble, list):
        raise TypeError("ensemble must be a list")

    # 28000
    img_cnt = len(dataset)
    # array(28000,1)
    predicted_labels = np.zeros(img_cnt, dtype=np.uint8)
    bs = 30
    loader = DataLoader(dataset, batch_size=bs)
    # array(10, 28000)
    predictions = np.zeros((len(ensemble), img_cnt))
    for j in range(len(ensemble)):
        model = ensemble[j]
        model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            i = 0
            for data in loader:
                data = data[0]
                # print(data.shape)
                data = model.adjustImages(data)
                data = data.to(device)
                output = model(data)

                # (bs, 2) tensor: rows contain 2 weighted probabilities
                # at idx==0: label is != j
                # at idx==1: label is == j
                output = func.softmax(output, 1).cpu()
                # array of values from 2nd column in output
                maxp = output.numpy()[:, 1]

                predictions[j][i * bs : i * bs + len(output)] = maxp
                i = i + 1
                # print(predicted_labels)
                # print(f"inferred {i*bs} from {len(images)}")
    predicted_labels = np.argmax(predictions, 0)
    print(f"predicted labels: {len(predicted_labels)}")
    return predicted_labels
