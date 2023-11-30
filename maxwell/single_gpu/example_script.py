'''Train CIFAR10 with PyTorch.'''
'''Source taken from https://github.com/kuangliu/pytorch-cifar and adapted by AidenDurrant'''

import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

# command line arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--epochs', default=200, type=int, help='epochs')
parser.add_argument('--save', default='~/sharedscratch', type=str, help='directory of the dataset')


best_acc = 0

def main():

    args = parser.parse_args()
    
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data Transforms (Augmentations)
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download the dataset
    trainset = torchvision.datasets.CIFAR10(
        root=args.save, train=True, download=True, transform=transform_train)
    # Construct dataset, dataloader for training split
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=args.save, train=False, download=True, transform=transform_test)
    # Construct dataset, dataloader for testing split
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False, num_workers=2)

    '''
    TO ENABLE GPU ACCELERATION SEND MODEL TO GPU IF AVAILABLE

    Check if GPU available:
    torch.cuda.is_available()

    Send model to the GPU (device='cuda')
    net.to(device)

    https://PyTorch.org/docs/stable/notes/cuda.html#best-practices
    '''
    # Check if GPUs are available, if so set device to CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialise the VGG model
    print('==> Building model..')
    net = VGG()
    net = net.to(device) # Send to CUDA aka GPU

    criterion = nn.CrossEntropyLoss().to(device) # Loss Function
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) # Optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) # Learning rate decay

    # Iterate over training epochs
    for epoch in range(start_epoch, start_epoch+args.epochs):
        train(epoch, trainloader, optimizer, criterion, net, device) # Run training loop
        scheduler.step()
    # Run test / evaluation
    test(epoch, testloader, criterion, net, device)

# Training
def train(epoch, trainloader, optimizer, criterion, net, device):
    print('\nEpoch: %d' % epoch)
    net.train() # set network to training

    # Initalize metrics
    train_loss = 0
    correct = 0
    total = 0

    # Iterate over all data samples
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device) # Send samples to device
        
        optimizer.zero_grad()
        
        outputs = net(inputs) # Forward pass
        loss = criterion(outputs, targets) # Loss
        loss.backward() # Compute grads
        
        optimizer.step() # Backwards pass

        # Store metrics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print metrics every 25 batches
        if batch_idx % 25 == 0:
            tqdm.write("[Train] \t Loss: {:.3f}\t Acc: {:.3f} ({}/{})".format(train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, testloader, criterion, net, device):
    global best_acc
    net.eval() # set network to evalutation

    # Initalize metrics
    test_loss = 0
    correct = 0
    total = 0

    # Iterate over all data samples with no gradients computed for evaluation
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)  # Send samples to device
            
            outputs = net(inputs) # Inference
            loss = criterion(outputs, targets) # Loss

            # Store metrics
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Print metrics every 25 batches
            if batch_idx % 25 == 0:
                tqdm.write("[Test] \t Loss: {:.3f}\t Acc: {:.3f} ({}/{})".format(test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    # Save checkpoint if current accuracy is the best
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

# VGG convolutional architecture https://arxiv.org/abs/1409.1556
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        net_conf = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'] # VGG 11
        self.features = self._make_layers(net_conf)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

if __name__ == '__main__':
    main()
