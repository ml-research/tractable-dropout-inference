#!/usr/bin/env python3

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time

import os
import sys

from torch import optim
from torchvision import datasets, transforms
from torch import nn

import pdb
import datetime

mean = 0.1307
std = 0.3081

n_classes = 10
transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean), (std))])

train_set = datasets.MNIST(root='../data', download=True, train=True, transform=transformer)
test_set = datasets.MNIST(root='../data', download=True, train=False, transform=transformer)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=200, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=200, shuffle=False)

# Layer details for the neural network
input_size = 784
# hidden_sizes = [128, 64]
hidden_sizes = [1024, 544]
output_size = 10
dropout_p = 0.8

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.Dropout(p=dropout_p),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.Dropout(p=dropout_p),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
                      # nn.LogSoftmax(dim=1))
print(model)
print("N of params: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)


#criterion = nn.NLLLoss()
# images, labels = next(iter(trainloader))
# images = images.view(images.shape[0], -1)

# logps = model(images.cuda())
# loss = criterion(logps, labels.cuda())

# print('Before backward pass: \n', model[0].weight.grad)

# loss.backward()

# print('After backward pass: \n', model[0].weight.grad)


# # Optimizers require the parameters to optimize and a learning rate
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# print('Initial weights - ', model[0].weight)

# images, labels = next(iter(trainloader))
# images.resize_(64, 784)

# # Clear the gradients, do this because gradients are accumulated
# optimizer.zero_grad()

# # Forward pass, then backward pass, then update weights
# output = model(images.cuda())
# loss = criterion(output, labels.cuda())
# loss.backward()
# print('Gradient -', model[0].weight.grad)

# # Take an update step and few the new weights
# optimizer.step()
# print('Updated weights - ', model[0].weight)

# TODO add early stopping
criterion = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.Adam(model.parameters(), lr=1e-3)

time0 = time()
epochs = 200 #200
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images.cuda())
        loss = criterion(output, labels.cuda())

        #This is where the model learns by backpropagating
        loss.backward()

        #And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_loader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)


def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.cpu().data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()



# images, labels = next(iter(valloader))

# img = images[0].view(1, 784)
# # Turn off gradients to speed up this part
# with torch.no_grad():
#     logps = model(img.cuda())

# # Output of the network are log-probabilities, need to take exponential for probabilities
# ps = torch.exp(logps)
# probab = list(ps.cpu().numpy()[0])
# print("Predicted Digit =", probab.index(max(probab)))
# view_classify(img.view(1, 28, 28), ps)

datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model.eval()
rotation_degrees = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 0]

for degrees in rotation_degrees:
    preds = np.empty((0, n_classes))
    print("Rotating test digits of {} degrees...".format(degrees))
    correct_count, all_count = 0, 0
    for images, labels in test_loader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            img = torchvision.transforms.functional.rotate(img.reshape(-1,1,28,28), degrees, fill=-mean/std).reshape(-1,28,28)
            img = img.view(1,784)
            # Turn off gradients to speed up this part
            with torch.no_grad():
                output = model(img.cuda())
                # logps = model(img.cuda())

                # Output of the network are log-probabilities, need to take exponential for probabilities
                # ps = torch.exp(logps)
                # probab = list(ps.cpu().numpy()[0])

                probab = list(output.cpu().numpy()[0])
                pred_label = probab.index(max(probab))
                true_label = labels.numpy()[i]
                if(true_label == pred_label):
                    correct_count += 1
                all_count += 1
                probab = np.array(probab)
                preds = np.vstack((preds, probab[np.newaxis, :]))

    print("Number of images tested =", all_count)
    print("Model Accuracy =", (correct_count/all_count))
    print("Average classification confidence {}".format(preds.max(axis=1).mean()))
    np.save('./predictions_nn_' + str(degrees) + '_' + datetime_str, preds)

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def get_monte_carlo_predictions(data_loader,
                                forward_passes,
                                model,
                                n_classes,
                                n_samples,
                                rotation=None):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """

    mean = 0.1307
    std = 0.3081

    dropout_predictions = np.empty((0, n_samples, n_classes))
    softmax = nn.Softmax(dim=1)
    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        model.eval()
        enable_dropout(model)
        for k, (image, label) in enumerate(data_loader):
            output = torch.zeros(len(label), n_classes)
            image = image.to(torch.device('cuda'))
            for j in range(len(label)):
                with torch.no_grad():
                    if rotation is None:
                        output[j] = model(image[j].view(1,784))[0]
                    else:
                        rotated_img = torchvision.transforms.functional.rotate(image[j].reshape(-1,1,28,28), rotation, fill=-mean/std).reshape(-1,28,28)
                        output[j] = model(rotated_img.view(1,784))[0]


                    # output = softmax(output) # shape (n_samples, n_classes)
            predictions = np.vstack((predictions, output.cpu().numpy()))

        # breakpoint()
        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))
        # dropout predictions - shape (forward_passes, n_samples, n_classes)

    # Calculating mean across multiple MCD forward passes
    mean = np.mean(dropout_predictions, axis=0) # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes
    variance = np.var(dropout_predictions, axis=0) # shape (n_samples, n_classes)

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes
    entropy = -np.sum(mean*np.log(mean + epsilon), axis=-1) # shape (n_samples,)

    # Calculating mutual information across multiple MCD forward passes
    mutual_info = entropy - np.mean(np.sum(-dropout_predictions*np.log(dropout_predictions + epsilon),
                                            axis=-1), axis=0) # shape (n_samples,)

    return dropout_predictions, mean, variance, entropy, mutual_info


fwd_passes = 100
datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


dropout_predictions, mean, variance, entropy, mutual_info =  get_monte_carlo_predictions(test_loader, fwd_passes, model, n_classes, test_loader.dataset.targets.size(dim=0), rotation=None)
print("Accuracy: {}".format((test_loader.dataset.targets.numpy()  == mean.argmax(axis=1)).sum()))
np.save('./dropout_predictions_nn_' + datetime_str, dropout_predictions)

for rotation in rotation_degrees:
    dropout_predictions, mean, variance, entropy, mutual_info =  get_monte_carlo_predictions(test_loader, fwd_passes, model, n_classes, test_loader.dataset.targets.size(dim=0), rotation=rotation)
    print("Accuracy on images rotated of {} degrees: ({}/{}) {}".format(rotation, (test_loader.dataset.targets.numpy()  == mean.argmax(axis=1)).sum(), mean.shape[0], (test_loader.dataset.targets.numpy()  == mean.argmax(axis=1)).sum()/mean.shape[0]))
    print("Average classification confidence {}".format(mean.max(axis=1).mean()))
    np.save('./dropout_predictions_nn_' + str(rotation) + '_' + datetime_str, dropout_predictions)




