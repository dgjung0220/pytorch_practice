# autoencoder with pytorch
# FashionMNIST

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets

class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train(model, train_loader, optimizer, log_interval):

    model.train()

    for batch_idx, (image, _) in enumerate(train_loader):
        image = image.view(-1, 28 * 28).to(DEVICE)
        target = image.view(-1, 28 * 28).to(DEVICE)
        optimizer.zero_grad()

        encoded, decoded = model(image)
        loss = criterion(decoded, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch : {} / [{}/{}({:.0f}%)]\t Train Loss : {:.6f}'.format(Epoch, batch_idx * len(image),
                                                                                     len(train_loader.dataset),
                                                                                     100. * batch_idx / len(
                                                                                         train_loader), loss.item()))

def evaluate(model, test_loader):

    model.eval()

    test_loss = 0
    real_image = []
    gen_image = []

    with torch.no_grad():
        for image, _ in test_loader:
            image = image.view(-1, 28 * 28).to(DEVICE)
            target = image.view(-1, 28 * 28).to(DEVICE)
            encoded, decoded = model(image)

            test_loss += criterion(decoded, target).item()
            real_image.append(image.to('cpu'))
            gen_image.append(decoded.to('cpu'))

    test_loss /= len(test_loader.dataset)
    return test_loss, real_image, gen_image

if __name__ == '__main__':

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

    print('PyTorch version: ', torch.__version__, ' Device: ', DEVICE)

    BATCH_SIZE = 32
    EPOCHS = 10

    train_dataset = datasets.FashionMNIST(root='data/FashionMNIST',
                                          train=True,
                                          download=True,
                                          transform=transforms.ToTensor())
    test_dataset = datasets.FashionMNIST(root='data/FashionMNIST',
                                         train=False,
                                         transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)

    model = AE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    print(model)

    for Epoch in range(1, EPOCHS + 1):

        train(model, train_loader, optimizer, log_interval= 200)
        test_loss, real_image, gen_image = evaluate(model, test_loader)

        print('\n[EPOCH: {}], \t Test Loss: {:.4f}'.format(Epoch, test_loss))

        f, a = plt.subplots(2, 10, figsize=(10, 4))

        for i in range(10):
            img = np.reshape(real_image[0][i], (28, 28))
            a[0][i].imshow(img, cmap='gray_r')
            a[0][i].set_xticks(())
            a[0][i].set_yticks(())

        for i in range(10):
            img = np.reshape(gen_image[0][i], (28, 28))
            a[1][i].imshow(img, cmap='gray_r')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())

        plt.show()