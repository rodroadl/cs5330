# Your name here and a short header
# Name: Gungyoem James Kim
# Project 5: Recognition using Deep Networks
# 1. Build and train a network to recognize digits
#   G. Test the network on new inputs

# import statements
import matplotlib.pyplot as plt
from firstABCDE import MyNetwork
import sys
import os
import torch
from torch import optim
from PIL import Image
import torchvision.transforms as transforms

# main function (ye,s it needs a comment too)
def main(argv):
    ### handle any command line arguments in argv

    ### Setting hyperparameters
    learning_rate = 0.01
    momentum = 0.5

    random_seed = 42
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    ### Preparing the dataset
    dp = "data/jnum/resize/"
    transform = transforms.Compose([transforms.PILToTensor()])
    example_list = []
    example_targets = [2 , 8, 2, 7, 1, 7, 1, 9, 7, 3, 4, 0, 5, 6, 0, 4]
    for fp in os.listdir(dp):
        image = Image.open(dp+fp)
        img_tensor = transform(image).type(torch.FloatTensor)
        example_list.append(img_tensor)
    example_data = torch.stack(example_list)
    
    ### Read the network and run it on the new data set
    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr = learning_rate, momentum=momentum)

    network_state_dict = torch.load("results/model.pth")
    network.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load("results/optimizer.pth")
    optimizer.load_state_dict(optimizer_state_dict)

    network.eval()
    with torch.no_grad():
        output = network(example_data)


    fig = plt.figure()
    correct = 0
    for i in range(16):
        pred = output.data.max(1, keepdim=True)[1][i].item()
        plt.subplot(4,4,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            pred
        ))
        plt.xticks([])
        plt.yticks([])
        if pred == example_targets[i]:
            correct += 1
    print(f"Accuracy: {round(correct/16,2)}")
    plt.show()
    

if __name__ == "__main__":
    main(sys.argv)