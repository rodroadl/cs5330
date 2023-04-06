# Your name here and a short header
# Name: GunGyeom James Kim
# Project 5: Recognition using Deep Networks
# 1. Build and train a network to recognize digits
#   A. Get the MNIST digit data set
#   B. Make your network code repeatable
#   C. Build a network model
#   D. Train the model
#   E. Save the network to a file

#import statements
import sys
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict

# class definitions
### C. Build a network model
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(1, 10, kernel_size = 5)), # A convolution layer with 10 5x5 filters
            ("pool1", nn.MaxPool2d(2)), # A max pooling layer with a 2x2 window 
            ("relu1", nn.ReLU()), # and a ReLU function applied.
            ('conv2',nn.Conv2d(10, 20, kernel_size = 5)), # A convolution layer with 20 5x5 filters
            ('drop', nn.Dropout2d(p = 0.5)), # A dropout layer with a 0.5 dropout rate (50%)
            ('pool2', nn.MaxPool2d(2)), # A max pooling layer with a 2x2 window 
            ('relu2', nn.ReLU()), # and a ReLU function applied
            ('flatten', nn.Flatten()), # A flattening operation 
            ('fc1', nn.Linear(320, 50)), # followed by a fully connected Linear layer with 50 nodes 
            ('relu3', nn.ReLU()), # and a ReLU function on the output
            ('fc2', nn.Linear(50,10)), # A final fully connected Linear layer with 10 nodes 
            ('log_softmax', nn.LogSoftmax()) # and the log_softmax function applied to the output.
        ]))
        
        

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        logits = self.model(x)
        return logits

# useful functions with a comment for each function
def train_network(epoch, network, optimizer, train_loader, log_interval, train_losses, train_counter, batch_size_train, save =False):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}({100.*batch_idx/len(train_loader):.0f}%)\tLoss: {loss.item():.6f}]')
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * batch_size_train)+ ((epoch-1) * len(train_loader.dataset)))
            
            
            ### E.Save the network to a file
            if save:
                torch.save(network.state_dict(), 'results/model.pth')
                torch.save(optimizer.state_dict(), 'results/optimizer.pth')
    return

def test_network(network, test_loader, test_losses):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%\n)'.format(
            test_loss, correct, len(test_loader.dataset),
            100.*correct / len(test_loader.dataset)))
    return
# main function (yes, it nees a comment too)
def main(argv):
    # handle any command line arguments in argv

    # main function code
    ### Preparing the Dataset
    n_epochs = 5
    batch_size_train=64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    ### B.Make your network code repeatable
    torch.manual_seed(42)
    torch.backends.cudnn.enabled = False

    ### A. Get the MNIST digit data set
    train_data = datasets.MNIST('/files/', train=True, download=True,
                       transform = Compose([ToTensor(), Normalize((0.1307,),(0.3081,))]))
    train_loader = DataLoader(train_data,batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(
        datasets.MNIST('/files/', train=False, download=True,
                       transform = Compose([ToTensor(), Normalize((0.1307,),(0.3081,))])), 
        batch_size=batch_size_test, shuffle=True)
    
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(train_data[i][0].permute(1,2,0), cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.show()

    ### Initialize the Network
    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    ### Training the Model
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs+1)]

    test_network(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
        train_network(epoch, network, optimizer, train_loader, log_interval, train_losses, train_counter, batch_size_train, save = True)
        test_network(network, test_loader, test_losses)

    ### Evaluating the Model's performance
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color = 'red')
    plt.legend(['Train Loss', 'Test Loss'], loc = 'upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()
    return

if __name__ == "__main__":
    main(sys.argv)
