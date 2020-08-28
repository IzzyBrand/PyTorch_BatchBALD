"""
Massachusetts Institute of Technology

Izzy Brand, 2020
"""
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
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(2048, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Dropout(0.1),
            nn.ReLU(),
        )


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.layers(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def active(model, aquirer, device, optimizer, num_batches=100):
    model.train()
    losses = []
    for batch_idx in range(num_batches):
        data, target = aquirer.select(model)
        #data, target = data.to(device), target.to(device)
        # the aquirer returned a single x, so we need make it into size-1 batch
        data, target = data[None,...].to(device), torch.LongTensor([target]).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Active: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data), num_batches,
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
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    num_pretrain = 1000
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
        batch_size=64, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,
        batch_size=64, pin_memory=True, shuffle=True)

    # init the model and optimizer with a learning-rate scheduler
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    # train the model and test after each epoch
    for epoch in range(1, 2):
        train(model, device, pretrain_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    pre_aquisition_model_state = model.state_dict()

    for aquisition_strategy in [Random, BALD]:
        # reset the model
        model.load_state_dict(pre_aquisition_model_state)
        # init the aquirer
        aquirer = aquisition_strategy(pool_data, device)
        # and an optimizer
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)
        # train the model
        losses = active(model, aquirer, device, optimizer)
        # plot the losses
        plt.plot(losses, label=aquisition_strategy.__name__)

    plt.legend()
    plt.show()
