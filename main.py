# PyTorch MLP Example

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets

import numpy as np
import matplotlib.pyplot as plt

def torch_init():
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    print('PyTorch version: ', torch.__version__, ' Device: ', DEVICE)

    return DEVICE

def set_datasets():
    train_dataset = datasets.MNIST(root='data/MNIST',
                                   train=True,
                                   download=True,
                                   transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='data/MNIST',
                                  train=False,
                                  transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader

def train(log_interval):

    model.train()

    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()

        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch : {} [ {}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(Epoch, batch_idx * len(image),
                                                                                   len(train_loader.dataset),
                                                                                   100. * batch_idx / len(train_loader),
                                                                                   loss.item()))

def evaluate():

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            output = model(image)
            test_loss += criterion(output, label).item()

            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, test_accuracy

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

if __name__ == '__main__':

    BATCH_SIZE = 32
    EPOCHS = 10

    DEVICE = torch_init()
    train_loader, test_loader = set_datasets()

    # Check the data
    # for (X_train, y_train) in train_loader:
    #     print('X_train : ', X_train.size(), 'type : ', X_train.type())
    #     print('y_train : ', y_train.size(), 'type : ', y_train.type())
    #     break

    # Visually check the data
    # plt.imshow(X_train[0,:,:,:].reshape(28,28), cmap='gray_r')
    # plt.title('Class : ' + str(y_train[0].item()))
    # plt.show()

    model = Net().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    print(model)

    for Epoch in range(1, EPOCHS + 1):
        # train(model, train_loader, optimizer, log_interval= 200)
        train(log_interval=200)
        test_loss, test_accuracy = evaluate()
        print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(Epoch, test_loss, test_accuracy))