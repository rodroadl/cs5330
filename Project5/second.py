# Your name here and a short header
# Name: Gungyeom James Kim
# Project 5: Recognition using Deep Networks
# 2. Examine your network
#   A. Analyze the first layer

# import statements
import matplotlib.pyplot as plt
from firstABCDE import MyNetwork
import sys
import torch
import torchvision
import cv2


# main function (yes, it needs a comment too)
def main(argv):
    ### handle any command line arguments in argv

    ### Setting hyperparameters
    random_seed = 42
    torch.backends.cudnn.enabled = False

    ### Read the network
    network = MyNetwork()
    network_state_dict = torch.load("results/model.pth")
    network.load_state_dict(network_state_dict)

    # Print the model
    print([mod for mod in network.modules() if not isinstance(mod, torch.nn.Sequential)])

    ### A. Analyze the first layer
    filters = []
    fig = plt.figure()
    for i in range(10):
        ndarr = network.model.conv1.weight[i,0].detach().numpy()
        filters.append(ndarr)
        plt.subplot(3, 4, i+1)
        plt.tight_layout()
        plt.imshow(ndarr, interpolation='none')
        plt.title(f"Filter {i}")
        plt.xticks([])
        plt.yticks([])
    plt.show()

    ### B. Show the effect of the filters
    # Get the first training example image
    train_data = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                       transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,),(0.3081))]))


    #
    with torch.no_grad():
        fig = plt.figure()
        for i in range(10):
            plt.subplot(5,4,2*i+1)
            plt.tight_layout()
            plt.imshow(filters[i], cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(5,4,2*i+2)
            plt.tight_layout()
            plt.imshow(cv2.filter2D(train_data[0][0].detach().numpy(), ddepth = -1, kernel = filters[i])[0], cmap='gray')
            plt.xticks([])
            plt.yticks([])
        plt.show()

if __name__ == "__main__":
    main(sys.argv)