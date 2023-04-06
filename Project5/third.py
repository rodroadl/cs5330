# Name: GunGyeom James Kim
# Project 5: Recognition using Deep Networks
# 3. Transfer Learning on Greek Letters

# import statements
import sys
import torch
import torchvision
from torchsummary import summary
from torch import nn
from torch import optim
from firstABCDE import MyNetwork, train_network, test_network

import matplotlib.pyplot as plt

# greek data set transform:
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28,28))
        return torchvision.transforms.functional.invert(x)

def main(argv):
    # set hyperparameters
    n_epochs = 1000
    learning_rate = 0.0001
    momentum = 0.9
    log_interval = 1

    torch.manual_seed(42)
    torch.backends.cudnn.enabled = False

    ### (1) generate the MNIST network
    network = MyNetwork()
    

    ### (2) read an existing model from a file and load the pre-trained weights
    network_state_dict = torch.load("results/model.pth")
    network.load_state_dict(network_state_dict)

    # optimizer_state_dict = torch.load("results/optimizer.pth")
    # optimizer.load_state_dict(optimizer_state_dict)

    ### (3) freeze the network weights
    # freezes the parameters for the whole network
    for param in network.parameters():
        param.requires_grad = False

    ### (4) replace the last layer with three nodes
    network.model.fc2 = nn.Linear(50, 3)
    
    params_to_update = [param for _, param in network.named_parameters() if param.requires_grad]
    optimizer = optim.SGD(params_to_update, lr = learning_rate, momentum=momentum)
    
    # printout of modified network
    print([mod for mod in network.modules() if not isinstance(mod, torch.nn.Sequential)])

    # DataLoader for the Greek data set
    batch_size_train = 27
    training_set_path = "data/greek_train"
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(training_set_path,
                                         transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                     GreekTransform(),
                                                                                     torchvision.transforms.Normalize(
                                                                                    (0.1307,),(0.3081,))])), 
                                        batch_size = batch_size_train, 
                                        shuffle = True)

    train_losses = []
    train_counter = []

    examples = enumerate(greek_train)
    _, (example_data, example_targets) = next(examples)

    for epoch in range(1, n_epochs+1):
        train_network(epoch, network, optimizer, greek_train, log_interval, train_losses, train_counter, batch_size_train)

    ### Evaluating the Model's performance
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    summary(network, (1,28,28))
    plt.show()

    letter = {0:"alpha", 1:'beta', 2:'gamma'}
    network.eval()
    with torch.no_grad():
        output = network(example_data)

    fig = plt.figure()
    n = len(output)
    for i in range(n):
        plt.subplot(6,6,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Pred: {}".format(
            letter[output.data.max(1, keepdim=True)[1][i].item()]
        ))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # Test
    batch_size_train = 8
    training_set_path = "data/greek_test"
    greek_test = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(training_set_path,
                                         transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                     GreekTransform(),
                                                                                     torchvision.transforms.Normalize(
                                                                                    (0.1307,),(0.3081,))])), 
                                        batch_size = batch_size_train, 
                                        shuffle = True)

    examples = enumerate(greek_test)
    _, (example_data, example_targets) = next(examples)

    network.eval()
    with torch.no_grad():
        output = network(example_data)

    fig = plt.figure()
    n = len(output)
    for i in range(n):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Pred: {}".format(
            letter[output.data.max(1, keepdim=True)[1][i].item()]
        ))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    return

if __name__ == "__main__":
    main(sys.argv)