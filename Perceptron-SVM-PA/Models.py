import numpy as np
import sys
from scipy import stats

my_hot_vector = {
    "M": np.asarray([0, 0, 1]).astype(np.float),
    "F": np.asarray([0, 1, 0]).astype(np.float),
    "I": np.asarray([1, 0, 0]).astype(np.float)
}


def create_tables(fileName, label=False):
    myData = np.genfromtxt(fileName, dtype=str, delimiter=",")
    if not label:
        gender = myData[:, 0]
        gender = np.asarray([my_hot_vector[s] for s in gender])
        myData = myData[:, 1:].astype(np.float)
        myData = np.asarray([np.concatenate((s, d), axis=None) for s, d in zip(gender, myData)])
    return myData.astype(np.float)


def create_y(fileName):
    Y_Train2 = []
    with open(fileName) as f2:
        Y_Train = [float(i.strip('\n')) for i in f2]
    for item in Y_Train:
        Y_Train2.append(int(item))
    return Y_Train2


def update_perceprton(x, y, w, eta):
    y_hat = np.argmax(np.dot(w, x))
    if y != y_hat:
        w[y, :] = w[y, :] + eta * x
        w[y_hat, :] = w[y_hat, :] - eta * x
    return w


def update_svm(x, y, w, eta, my_lambda):
    y_hat = np.argmax(np.dot(w, x))
    if y != y_hat:
        w[y, :] = (1 - eta * my_lambda) * w[y, :] + eta * x
        w[y_hat, :] = w[y_hat, :] - eta * x
    else:
        for i in range(0, 3):
            if i != y:
                w[i, :] = (1 - eta * my_lambda) * w[i, :]
    return w


def update_pa(x, y, w):
    norm = (np.linalg.norm(x)) ** 2
    y_hat = np.argmax(np.dot(w, x))
    if y != y_hat:
        a = 1 - np.dot(w[y, :], x) + np.dot(w[y_hat, :], x)
        loss = max(0, a)
        t = loss / norm
        w[y, :] = w[y, :] + t * x
        w[y_hat, :] = w[y_hat, :] - t * x
    return w


def min_max_normalization(arr):
    transArr = np.transpose(arr)
    for i in range(len(transArr)):
        transArr[i] = (transArr[i] - min(transArr[i])) / (max(transArr[i]) - min(transArr[i]))
    return np.transpose(transArr)


def z_score_normalization(arr):
    transArr = np.transpose(arr)
    transArr = stats.zscore(transArr)
    return np.transpose(transArr)


def training(epochs, algo):
    X_train = create_tables(sys.argv[1])
    Y_train = create_y(sys.argv[2])
    w = np.random.uniform(-0.5, 0.5, size=(3, 10))
    X_train = z_score_normalization(X_train)
    eta = 0.01
    my_lambda = 0.075
    for e in range(epochs):
        for x, y in zip(X_train, Y_train):
            if algo == "perceptron":
                eta = eta * 0.9999
                w = update_perceprton(x, y, w, eta)
            elif algo == "svm":
                eta = eta * 0.9999
                w = update_svm(x, y, w, eta, my_lambda)
            elif algo == "pa":
                w = update_pa(x, y, w)
    return w


def testing():
    X_test = create_tables(sys.argv[3])
    w1 = training(10, "perceptron")
    w2 = training(10, "svm")
    w3 = training(10, "pa")
    X_test = z_score_normalization(X_test)
    for x in X_test:
        y_hat1 = np.argmax(np.dot(w1, x))
        y_hat2 = np.argmax(np.dot(w2, x))
        y_hat3 = np.argmax(np.dot(w3, x))
        print("perceptron:", y_hat1, end="")
        print(", svm:", y_hat2, end="")
        print(", pa:", y_hat3)


testing()