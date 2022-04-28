import os
#os.chdir(r'D:\yaoli\tangent')
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)



def create_model(args):
    autoencoder = Autoencoder(args.dim)
    if args.model_load:
        print("Loading pre-trained models {}".format(args.model_load))
        state_dict = torch.load(os.path.join(args.model_folder,args.model_load))
        autoencoder.load_state_dict(state_dict)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder


def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Autoencoder(nn.Module):
    def __init__(self, dim=20):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
        )
        self.fc1 = nn.Linear(384, dim)
        
        self.fc2 = nn.Linear(dim, 384)
        self.decoder = nn.Sequential(
             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )
    
    def decode(self, batch_size, encoded):
        y = self.fc2(encoded)
        y = y.view((batch_size, 96, 2, 2))
        decoded = self.decoder(y)        
        return decoded

    def encode(self, x):
        x = self.encoder(x)
        encoded = x.view(x.size(0), -1)
        encoded = self.fc1(encoded)    
        return encoded
    
    def forward(self, x):
        x = self.encoder(x)
        encoded = x.view(x.size(0), -1)
        encoded = self.fc1(encoded)
        y = self.fc2(encoded)
        y = y.view(x.shape)
        decoded = self.decoder(y)
        return encoded, decoded


def train_epoch(epoch, autoencoder, trainloader, optimizer, criterion):
    epoch_loss = 0.0
    for i, (inputs, _) in enumerate(trainloader, 0):
        inputs = inputs.cuda()

        # ============ Forward ============
        _, outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        # ============ Backward ============
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ============ Logging ============
        epoch_loss += loss.data
    print('Epoch %d, loss: %.3f' %
                  (epoch + 1, epoch_loss / (i+1)))    
    return autoencoder, epoch_loss/(i+1)


def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    parser.add_argument("--save_model", action="store_false", default=True,
                        help="Whether to save the model trained.")
    parser.add_argument("--root", default=r'D:\yaoli',
                        help="Path to the root folder that contains data folder.")
    parser.add_argument("--model_folder", default='./models',
                        help="Path to the folder that contains checkpoint.")
    parser.add_argument("--model_load", default='ae_loss0.589.pt',
                        help="name of the model to load.")
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--save_thrd', type=float, default=0.6)
    parser.add_argument("--train_shuffle", action="store_false",  default=True,
                        help="shuffle in training or not")    
    args = parser.parse_args()

    # Create model
    autoencoder = create_model(args)

    # Load data
    transform = transforms.Compose(
                        [transforms.ToTensor()]
                        )
    trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.root,'data'), train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True, num_workers=1)
    testset = torchvision.datasets.CIFAR10(root=os.path.join(args.root,'data'), train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=1)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.valid:
        print("Loading checkpoint...")
        autoencoder.load_state_dict(torch.load(os.path.join(args.model_folder,args.model_load)))
        dataiter = iter(testloader)
        images, labels = dataiter.next()
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
        imshow(torchvision.utils.make_grid(images))

        images = Variable(images.cuda())

        decoded_imgs = autoencoder(images)[1]
        imshow(torchvision.utils.make_grid(decoded_imgs.data))

        exit(0)

    # Define an optimizer and criterion
    criterion = nn.BCELoss()
    
    lr = args.lr
    save_thrd = args.save_thrd
    optimizer = optim.Adam(autoencoder.parameters(),lr=lr)
    for epoch in range(args.n_epochs):
        # if (epoch+1) % 20 == 0:
        #     lr = lr/2
        # optimizer = optim.Adam(autoencoder.parameters(),lr=lr)
        autoencoder, avg_loss = train_epoch(epoch, autoencoder, trainloader, optimizer, criterion)
        if avg_loss < save_thrd:
            print("Loss smaller than {:.4f}, Save check point for epoch {}".format(save_thrd, epoch))
            torch.save(autoencoder.state_dict(), os.path.join(args.model_folder,"ae_loss{:.3f}.pt".format(avg_loss)))
            save_thrd = avg_loss
         
    print('Finished Training')
    print('Saving Model...')
    if not os.path.exists(args.model_folder):
        os.mkdir(args.model_folder)
    if args.save_model:
        torch.save(autoencoder.state_dict(), os.path.join(args.model_folder,"ae_loss{:.3f}.pt".format(avg_loss)))


if __name__ == '__main__':
    main()
