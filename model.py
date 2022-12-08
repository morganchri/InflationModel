import numpy as np

def polynomial_transform(x, degree):
    x_copy = x.copy()
    for i in range(2, degree + 1):
        x = np.concatenate((x, [j ** i for j in x_copy]), axis=1)
    return x

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def f(weights, x):
    return (weights[5] * x ** 5) + \
        (weights[4] * x ** 4) + \
        (weights[3] * x ** 3) + \
        (weights[2] * x ** 2) + \
        (weights[1] * x) + \
        weights[0]


def loss_mse(y, y_bar):
    return sum((y - y_bar) ** 2) / len(y)


def nonlinear_gradient(weights, x, y, lr):
    y = y.reshape(y.shape[0], 1)
    y_bar = np.array([f(weights, x_val) for x_val in x])
    y_bar = y_bar.reshape(y_bar.shape[0], 1)
    x_all = polynomial_transform(x.reshape(x.shape[0], 1), 5)
    x_all = np.concatenate((np.ones((x.shape[0], 1)), x_all), axis=1)
    w = x_all * (y - y_bar)
    n = y.shape[0]
    gradient = (-2 / n) * (w.sum(axis=0))
    new_x = weights - (lr * gradient)
    new_model_weights = new_x
    new_y_bar = np.array([f(new_model_weights, x_val) for x_val in x])
    updated_model_loss = loss_mse(y, new_y_bar)
    return updated_model_loss, new_model_weights, new_y_bar


def nonlinear_gradient_descent(weights, x, y, lr, epochs):
    loss_history = []
    betas = weights
    for i in range(epochs):
        loss = nonlinear_gradient(betas, x, y, lr)
        betas = loss[1]
        loss_history.append(loss[0])
        if (i > 0):
            loss_pct = ((loss_history[i] - loss_history[i - 1]) / loss_history[i - 1])
            if ((loss_pct.all()) <= 0.01):
                print("Hit Threshold")
                break
    return betas

def jacobian(x):
    return np.array([
        -x ** 0,
        -x,
        -x ** 2,
        -x ** 3,
        -x ** 4
    ]).T

def gauss_newton():
    return
