import numpy as np
import scipy.linalg as la
from scipy.optimize import curve_fit


def nonlinear_transform(x, degree):
    '''
    Transforms the data into nth degree data.  Takes on a mx1 numpy array and
    transforms it into a mxdegree numpy array iteratively.

    :param x: The data to be transformed. in the form of a mx1 numpy array
    :param degree: The highest degree wanted
    :return: The new data in the form of a mxdegree numpy array
    '''
    x_copy = x.copy()
    for i in range(2, degree + 1):
        x = np.concatenate((x, [j ** i for j in x_copy]), axis=1)
    return x


def normalize(x):
    '''
    Min-max nomalization equation.

    :param x: The input to be normalized
    :return: Normalized data
    '''
    return (x - x.min()) / (x.max() - x.min())


def f(weights, x):
    '''
    Function for evaluating y using matrix multiplication

    :param weights: The weights, or coefficients for the equation
    :param x: The input data used to evaluate y
    :return: a mx1 numpy array of estimated y values
    '''
    return weights.dot(x.T)


def loss_mse(y, y_bar):
    return sum((y.sum(axis=0) - y_bar.sum(axis=0)) ** 2) / len(y)


def nonlinear_gradient(weights, x, y, lr):
    y_bar = f(weights, x)
    y_bar = y_bar.reshape(y_bar.shape[0], 1)
    w = x * (y - y_bar)
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
    return betas, loss_history


def jacobian(x):
    return np.array([-x, -x ** 2, -x ** 3, -x ** 4, -x ** 5])


def residual(weights, x, y):
    y_bar = f(weights, x)
    y_bar = y_bar.reshape(y_bar.shape[0], 1)
    return y - y_bar


def gauss_newton(weights, x, y):
    J = jacobian(x[:, 0]).T
    r = -residual(weights, x, y).reshape(1, -residual(weights, x, y).shape[0])[0]
    return weights + la.lstsq(J, r)[0]


def function(x, a, b, c, d, e, f):
    return (a * x ** 5) + (b * x ** 4) + (c * x ** 3) + (d * x ** 2) + (e * x) + f


def lm(f, x, y, weights):
    popt, pcov = curve_fit(f, x, y, p0=weights)
    return popt
