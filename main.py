"""
Massachusetts Institute of Technology

Izzy Brand, 2020
"""
from copy import deepcopy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from aquisition import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(784, 2048),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(2048, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 10))

    def forward(self, x, return_logits=False):
        x = torch.flatten(x, 1)
        logits = self.layers(x)
        if return_logits:
            return logits
        else:
            return F.softmax(logits, 1)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, return_logits=True)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def active(model, aquirer, device, optimizer, num_batches=100):
    batch_size = 5
    model.train()
    losses = []
    for batch_idx in range(num_batches):
        data, target = aquirer.select_batch(model, batch_size)
        #data, target = data.to(device), target.to(device)
        # the aquirer returned a single x, so we need make it into size-1 batch
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, return_logits=True)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Active {}:\t[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                aquirer.__class__.__name__, batch_idx * len(data), num_batches*batch_size,
                100. * batch_idx / num_batches, loss.item()))

        losses.append(loss.item())

    return losses

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, return_logits=True)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    lr = 0.001
    batch_size = 64
    num_pretrain = 10000
    num_pool = 1000
    num_extra = 60000 - num_pretrain - num_pool

    # set up the GPU if one exists
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load the dataset and pre-process
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
    pretrain_data, pool_data, _ =torch.utils.data.random_split(
        dataset, [num_pretrain, num_pool, num_extra])
    test_data = datasets.MNIST('data', train=False,
                       transform=transform)
    pretrain_loader = torch.utils.data.DataLoader(pretrain_data,
        batch_size=batch_size, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,
        batch_size=batch_size, pin_memory=True, shuffle=True)

    # init the model and optimizer
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train the model and test after each epoch
    for epoch in range(1, 2):
        train(model, device, pretrain_loader, optimizer, epoch)
        test(model, device, test_loader)\

    pre_aquisition_model_state = model.state_dict()

    for aquisition_strategy in [Random, BatchBALD, BALD]:
        # reset the model
        model.load_state_dict(deepcopy(pre_aquisition_model_state))
        # init the aquirer
        aquirer = aquisition_strategy(pool_data, device)
        # and an optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # train the model
        losses = active(model, aquirer, device, optimizer)
        # plot the losses
        plt.plot(losses, label=aquisition_strategy.__name__)

    plt.legend()
    plt.show()
