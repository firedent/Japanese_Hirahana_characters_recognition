#!/usr/bin/env python3
"""
part3_test.py

UNSW COMP9444 Neural Networks and Deep Learning

ONLY COMPLETE METHODS AND CLASSES MARKED "TODO".

DO NOT MODIFY IMPORTS. DO NOT ADD EXTRA FUNCTIONS.
DO NOT MODIFY EXISTING FUNCTION SIGNATURES.
DO NOT IMPORT ADDITIONAL LIBRARIES.
DOING SO MAY CAUSE YOUR CODE TO FAIL AUTOMATED TESTING.
"""
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import numpy as np

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# dev = torch.device("cpu")
class Linear(nn.Module):
    """
    DO NOT MODIFY
    Linear (10) -> ReLU -> LogSoftmax
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1).to(dev) # make sure inputs are flattened

        x = F.relu(self.fc1(x))
        x = F.log_softmax(x, dim=1)  # preserve batch dim

        return x


class FeedForward(nn.Module):
    """
    Linear (256) -> ReLU -> Linear(64) -> ReLU -> Linear(10) -> ReLU-> LogSoftmax
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1).to(dev)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = F.log_softmax(x, dim=1)
        return x


class CNN(nn.Module):
    """
    conv1 (channels = 10, kernel size= 5, stride = 1) -> Relu -> max pool (kernel size = 2x2) ->
    conv2 (channels = 50, kernel size= 5, stride = 1) -> Relu -> max pool (kernel size = 2x2) ->
    Linear (256) -> Relu -> Linear (10) -> LogSoftmax


    Hint: You will need to reshape outputs from the last conv layer prior to feeding them into
    the linear layers.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1).to(dev)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=50, kernel_size=5, stride=1).to(dev)
        self.fc1 = nn.Linear(50 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.shape).to(dev)

        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = F.log_softmax(x, dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class NNModel:
    def __init__(self, network, learning_rate):
        """
        Load Data, initialize a given network structure and set learning rate
        DO NOT MODIFY
        """

        # Define a transform to normalize the data
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

        # Download and load the training data
        trainset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

        # Download and load the test data
        testset = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

        self.model = network
        self.model.to(dev)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.num_train_samples = len(self.trainloader)
        self.num_test_samples = len(self.testloader)

        """
        TODO: Set appropriate loss function such that learning is equivalent to minimizing the
        cross entropy loss. Note that we are outputting log-softmax values from our networks,
        not raw softmax values, so just using torch.nn.CrossEntropyLoss is incorrect.
        
        Hint: All networks output log-softmax values (i.e. log probabilities or.. likelihoods.). 
        """
        self.lossfn = F.nll_loss

    def view_batch(self):
        """
        Do not make calls to plt.imshow() here

        Return:
           1) A float32 numpy array (of dim [28*8, 28*8]), containing a tiling of the batch images,
           place the first 8 images on the first row, the second 8 on the second row, and so on

           2) An int 8x8 numpy array of labels corresponding to this tiling
        """
        for i in self.trainloader:
            first_batch = i[0].reshape(8, 8, 28, 28)
            first_batch = torch.cat([_ for _ in torch.stack([_ for _ in first_batch], 1)], 2).flatten(0, 1)
            labels = i[1].reshape(8, 8).numpy()
            first_batch = first_batch.numpy()
            return first_batch, labels

    def train_step(self):
        """
        Used for submission tests and may be usefull for debugging
        DO NOT MODIFY
        """
        self.model.train()
        for images, labels in self.trainloader:
            log_ps = self.model(images)
            loss = self.lossfn(log_ps, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return

    def train_epoch(self):
        self.model.train()
        for images, labels in self.trainloader:
            log_ps = self.model(images)
            loss = self.lossfn(log_ps, labels.to(dev))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return

    def eval(self):
        self.model.eval()
        accuracy = 0
        with torch.no_grad():
            for images, labels in self.testloader:
                log_ps = self.model(images.to(dev))
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.to(dev).view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        return accuracy / self.num_test_samples


def plot_result(results, names):
    """
    Take a 2D list/array, where row is accuracy at each epoch of training for given model, and
    names of each model, and display training curves
    """
    for i, r in enumerate(results):
        plt.plot(range(len(r)), r, label=names[i])
    plt.legend()
    plt.title("KMNIST")
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("./part_2_plot.png")


def main():
    models = [Linear(), FeedForward(), CNN()]  # Change during development
    epochs = 10
    results = []

    # Can comment the below out during development
    images, labels = NNModel(Linear(), 0.003).view_batch()
    print(labels)
    plt.imshow(images, cmap="Greys")
    plt.show()

    for model in models:
        print(f"Training {model.__class__.__name__}...")
        m = NNModel(model, 0.003)

        accuracies = [0]
        for e in range(epochs):
            m.train_epoch()
            accuracy = m.eval()
            print(f"Epoch: {e}/{epochs}.. Test Accuracy: {accuracy}")
            accuracies.append(accuracy)
        results.append(accuracies)

    plot_result(results, [m.__class__.__name__ for m in models])


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(t2-t1)