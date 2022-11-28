import numpy as np


def cost_function(theta, x, y):
    m = len(y)
    pred = x.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(pred - y))
    return cost


def polynomial_transform(x, degree):
    x_copy = x.copy()

    for i in range(degree - 1, degree + 1):
        x = np.concatenate((x, [j ** i for j in x_copy]), axis=1)
    print(x)
    return x


def forward_n(x, w, b):
    return np.dot(x, w) + b


def loss(y, y_pred):
    return np.sqrt(np.mean((y_pred - y) ** 2))


def gradient(x, y, y_pred):
    m = x.shape[0]
    dw = (2 / m) * np.dot(x.T, (y_pred - y))
    db = (2 / m) * np.sum((y_pred - y))
    return dw, db


def polynomial_gradient_descent(x, y, epochs, lr, degree, display=True):
    x_new = polynomial_transform(x, degree)
    n = x_new.shape[1]

    #weights = np.random.random((n, 1))
    weights = np.zeros((n, 1))
    bias = 0

    loss_history = []
    for epoch in range(epochs):
        losses = []
        y_pred = forward_n(x_new, weights, bias)
        dw, db = gradient(x_new, y, y_pred)
        weights = weights - lr * dw
        bias = bias - lr * db
        y_pred = forward_n(x_new, weights, bias)
        l = loss(y, y_pred)
        losses.append(l)
        loss_history.append(l)
        if ((epoch % 10 == 0 or epoch == epochs - 1) and display == True):
            print("progress:", epoch, "loss=", np.mean(losses))
    return weights, bias, loss_history


def predict(x, w, b, degree):
    x = polynomial_transform(x, degree)
    return np.dot(x, w) + b

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def linear_gradient_descent(x, y, theta, l=0.03, iterations=100):
    m = len(y)
    c_hist = np.zeros(iterations)
    t_hist = np.zeros((iterations, 2))
    for i in range(iterations):
        pred = np.dot(x, theta)
        theta -= (1 / m) * l * (x.T.dot((pred - y)))
        t_hist[i, :] = theta.T
        c_hist[i] = cost_function(theta, x, y)
    return theta, c_hist, t_hist


def gauss_newton():
    return
