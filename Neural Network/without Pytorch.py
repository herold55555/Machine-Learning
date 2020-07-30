import numpy as np
import scipy


def relU(z):
    return np.max(0, z)


def normalization(table):
    return (table - 128) / 128


def initialize_weights(size1, size2):
    w = (np.random.rand(size1, size2) - .5) * .1
    return w


def loadData(filename):
    data = np.loadtxt(filename)
    return data


def softmax(x):
    # Compute the softmax of vector x.
    exps = np.exp(x)
    return exps / np.sum(exps)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def NLL(y_hat, y):
    loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return loss


def forward(x, parameters, active_func):
    w1, b1, w2, b2 = [parameters[key] for key in ('w1', 'b1', 'w2', 'b2')]
    z1 = np.dot(x, w1.T) + b1
    h = active_func(z1)
    z2 = np.dot(h, w2.T) + b2
    y_hat = softmax(z2)
    ret = {'y_hat': y_hat, 'h': h, 'z1': z1}
    for key in parameters:
        ret[key] = parameters[key]
    return ret


def backPropagation(forwardData, x, y, active_function):
    # receive all the data and calculate all the derivative for the gradients
    y_hat, h, z1, w1, b1, w2, b2 = [forwardData[key] for key in ('y_hat', 'h', 'z1', 'w1', 'b1', 'w2', 'b2')]
    # deltaLoss / z2
    deltaZ2 = y_hat
    deltaZ2[0][int(y)] -= 1
    gradient_w2 = np.dot(deltaZ2.T, h)
    gradient_b2 = deltaZ2
    # deltaLoss / H
    deltaH = np.dot(deltaZ2, w2)
    sig = active_function(z1) * (1 - active_function(z1))
    # deltaLoss / deltaZ1
    deltaZ1 = deltaH * sig
    gradient_w1 = np.dot(deltaZ1.T, x)
    gradient_b1 = deltaZ1
    gradient = {'gradW1': gradient_w1, 'gradB1': gradient_b1, 'gradW2': gradient_w2, 'gradB2': gradient_b2}
    return gradient


def update_weights_sgd(params, gradients, eta):
    # receive the old data
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    # update the data : Wi = W(i-1) - learning rate * deltaLoss / deltaW‬ ‬
    newW1 = w1 - eta * gradients['gradW1']
    newW2 = w2 - eta * gradients['gradW2']
    newB1 = b1 - eta * gradients['gradB1']
    newB2 = b2 - eta * gradients['gradB2']
    # return dictionary with the updated values
    update = {'w1': newW1, 'w2': newW2, 'b1': newB1, 'b2': newB2}
    return update


def train(params, epochs, active_func, lr, train_x, train_y):
    for i in range(epochs):  # for each epoch:
        sum_loss = 0.0
        for x1, y in zip(train_x, train_y):
            x = np.reshape(x1, (1, 784))
            out = forward(x, params, active_func)
            y_hat = out['y_hat']
            loss = NLL(y_hat, y)  # compute loss to see train loss (for hyper parameters tuning)
            sum_loss += loss
            gradients = backPropagation(out, x, y, active_func)  # returns the gradients for each parameter
            params = update_weights_sgd(params, gradients, lr)  # updates the weights
        # after each epoch, check accuracy and loss on dev set for hyper parameter tuning
    return params


def main():
    x_train = loadData("train_x")
    y_train = loadData("train_y")
    x_test = loadData("test_x")
    x_train = normalization(x_train)
    x_test = normalization(x_test)
    hidden_size = 50
    eta = 0.01
    epochs = 80
    sizeOfX = 784
    sizeofY = 10
    w1 = initialize_weights(hidden_size, sizeOfX)
    b1 = initialize_weights(1, hidden_size)
    w2 = initialize_weights(sizeofY, hidden_size)
    b2 = initialize_weights(1, sizeofY)
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    parameters = train(parameters, epochs, sigmoid, eta, x_train, y_train)
    f = open("test_y", "w")
    for x in x_test:
        x = np.reshape(x, (1, 784))
        out = forward(x, parameters, sigmoid)
        y_hat = out['y_hat']
        f.write(str(y_hat.argmax()) + '\n')
    f.close()


main()