#Source: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
import matplotlib.pyplot as plt

import os

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "/home/raja/Documents/raja/torchexp/standard_googlenet.pth"
use_cuda=True
#datadir = "/home/raja/Documents/raja/torchexp/data"

vtype_train = datasets.ImageFolder(os.path.join("/home/raja/Documents/raja/torchexp/data/train"), transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
vtype_test = datasets.ImageFolder(os.path.join("/home/raja/Documents/raja/torchexp/data/val"), transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
train_loader = DataLoader(vtype_train, batch_size = 96, shuffle=True)
test_loader = DataLoader(vtype_test, batch_size = 1, shuffle=True)
# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

Net = models.googlenet(num_classes =5)

num_ftrs = Net.aux1.fc2.in_features
Net.aux1.fc2 = nn.Linear(num_ftrs, 5)
        
num_ftrs = Net.aux2.fc2.in_features
Net.aux2.fc2 = nn.Linear(num_ftrs, 5)
        # Handle the primary net
        #num_ftrs = model_ft.fc.in_features
        #model_ft.fc = nn.Linear(num_ftrs,num_classes)
num_ftrs = Net.fc.in_features
Net.fc = nn.Linear(num_ftrs, 5)

# Initialize the network
model = Net.to(device)
#model.eval()

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
#model.eval()


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, -2.117, 2.64)
    #perturbed_image = Variable(perturbed_image.data_grad)
    #images_adv = Variable(images_adv.data,requires_grad = True)
    # Return the perturbed image
    return perturbed_image


def test( model, device, test_loader, epsilon ):
    
    model.eval()

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        #print('target shape:',target.shape)
        #print('data shape', data.shape)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]# get the index of the max log-probability
        #print('init_pred shape:',init_pred.shape)

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = criterion(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data
        #print(data_grad.shape)

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        #print(perturbed_data.shape)

        # Re-classify the perturbed image
        output = model(perturbed_data)
        #print(output.shape)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        #print(final_pred.shape)
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 6):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                #print(adv_ex.shape)
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 6:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

criterion = nn.CrossEntropyLoss()
accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)


plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.savefig("standdard_eps_vs_acc_plot.png")
#plt.show()


# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        ex = ex.transpose((1, 2, 0))
        mapping = {0: 'bus', 1: 'pickup', 2: 'sedam', 3: 'truck', 4: 'van'}
        plt.title("{} -> {}".format(mapping[orig], mapping[adv]))
        plt.imshow(ex)
plt.tight_layout()
plt.savefig("standard_vis_fgsm.pdf")
#plt.show()
