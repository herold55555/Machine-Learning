import numpy as np
import torch
from torch import nn, optim
from torch.version import cuda

from gcommand_loader import GCommandLoader


def loadData(path, size):
    x1 = torch.utils.data.DataLoader(
        path, batch_size=size, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)
    return x1


def getDictionnary():
    labels = dict()
    labels.__setitem__("bed", 0), labels.__setitem__("bird", 1), labels.__setitem__("cat", 2), labels.__setitem__(
        "dog", 3),labels.__setitem__("down", 4),
    labels.__setitem__("eight", 5), labels.__setitem__("five", 6), labels.__setitem__("four", 7), labels.__setitem__(
        "go", 8),
    labels.__setitem__("happy", 9), labels.__setitem__("house", 10), labels.__setitem__("left", 11), labels.__setitem__(
        "marvin", 12),
    labels.__setitem__("nine", 13), labels.__setitem__("no", 14), labels.__setitem__("off", 15), labels.__setitem__("on",
                                                                                                                   16), labels.__setitem__(
        "one", 17)
    labels.__setitem__("right", 18), labels.__setitem__("seven", 19), labels.__setitem__("sheila",
                                                                                         20), labels.__setitem__("six",
                                                                                                                 21), labels.__setitem__(
        "stop", 22),
    labels.__setitem__("three", 23), labels.__setitem__("tree", 24), labels.__setitem__("two", 25), labels.__setitem__(
        "up", 26), labels.__setitem__("wow", 27),
    labels.__setitem__("yes", 28), labels.__setitem__("zero", 29)
    return labels


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(1, 30, kernel_size=4, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2))
        #self.layer2 = nn.Sequential(
        #nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
        #nn.ReLU(),
        #nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(80 * 50 * 30, 1000)
        self.fc2 = nn.Linear(1000, 30)

    def forward(self, x):
        out = self.layer1(x)
        #out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def validation_model(self, set_validation):
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for examples, labels in set_validation:
                outputs = self(examples)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print("accuracy : {} %".format((correct/total)*100))

def main():

    dataSetTest = GCommandLoader("/home/herold55/PycharmProjects/ex4ML/Test")
    dataSetTrain = GCommandLoader("/home/herold55/PycharmProjects/ex4ML/Train2")
    dataSetValid = GCommandLoader("/home/herold55/PycharmProjects/ex4ML/Valid")
    print(len(dataSetTest))
    print(len(dataSetTrain))
    print(len(dataSetValid))
    x_test = loadData(dataSetTest, 100)
    x_train = loadData(dataSetTrain, 100)
    x_valid = loadData(dataSetValid, 100)
    epochs = 5
    learning_rate = np.exp(-23)
    model = ConvNet()
    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    total_step = len(x_train)
    loss_list = []
    acc_list = []
    for epoch in range(epochs):
        for (images, labels) in x_train:
            # Run the forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

        #if (True):
         #   print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
          #        .format(epoch + 1, epochs, total_step, loss.item(),
           #               (correct / total) * 100))
    model.validation_model(x_valid)

main()
