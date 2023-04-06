# Your name here and a short header
# Name: Gungyoem James Kim
# Project 5: Recognition using Deep Networks
# 1. Build and train a network to recognize digits
#   F. Read the network and run it on the test set

# import statements
import matplotlib.pyplot as plt
from firstABCDE import MyNetwork
import sys
import torch
import torchvision
from torch import optim

# main function (yes, it needs a comment too)
def main(argv):
    ### handle any command line arguments in argv

    ### Setting hyperparameters
    learning_rate = 0.01
    momentum = 0.5
    batch_size_test = 10
    
    random_seed = 42
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    ### Preparing the dataset
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,),(0.3081,))
                                   ])), batch_size = batch_size_test, shuffle = True
    )

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    ### F. Read the network and run it on the test set
    continued_network = MyNetwork()
    continued_optimizer = optim.SGD(continued_network.parameters(), lr = learning_rate, momentum = momentum)

    network_state_dict = torch.load("results/model.pth")
    continued_network.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load("results/optimizer.pth")
    continued_optimizer.load_state_dict(optimizer_state_dict)

    continued_network.eval()
    with torch.no_grad():
        output = continued_network(example_data)
    
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()
        ))
        print(f"For example {i}:")
        print("\t10 output values:", [round(num.item(), 2) for num in output.data[i]])
        print("\tThe index of the max output value:", output.data[i].argmax().item())
        print("\tThe correct label of digit:", example_targets[i].item())
        plt.xticks([])
        plt.yticks([])
    print(f"For example {9}:")
    print("\t10 output values:", [round(num.item(), 2) for num in output.data[9]])
    print("\tThe index of the max output value:", output.data[9].argmax().item())
    print("\tThe correct label of digit:", example_targets[9].item())
    plt.show()

if __name__ == "__main__":
    main(sys.argv)