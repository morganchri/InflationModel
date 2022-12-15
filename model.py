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
    '''
    Loss function.  Returns the loss

    :param y: numpy Array representing the existing y values
    :param y_bar: numpy Array representing the calculateed y values
    :return: float representing the mean of the squared error
    '''
    return ((y - y_bar) ** 2).mean()


def nonlinear_gradient(weights, x, y, lr):
    '''
    Calculates the gradient, new weights, new loss, and new y_bar for the gradient descent algorithm,
    is what is being iterated over during the Gradient Descent algorithm

    :param weights: numpy array of the weights that are being adjusted
    :param x: numpy array of input parameters
    :param y: numpy array of the output parameters
    :param lr: learning rate for the algorithm
    :return: the new loss, new weights, and new estimated y values
    '''
    y_bar = f(weights, x)
    y_bar = y_bar.reshape(y_bar.shape[0], 1)
    w = x * (y - y_bar)
    n = y.shape[0]
    gradient = (-2 / n) * (w.sum(axis=0))
    new_w = weights - (lr * gradient)
    new_model_weights = new_w
    new_y_bar = f(new_model_weights, x)
    updated_model_loss = loss_mse(y, new_y_bar)
    return updated_model_loss, new_model_weights, new_y_bar


def nonlinear_gradient_descent(weights, x, y, lr, epochs):
    '''
    Iterate across a set number of epochs given an initial set of weights, x inputs, y outputs, and learning rate.

    :param weights: initial set of weights to be adjusted for the gradient descent algorithm, typically a numpy array
        of random values from 0 to 1
    :param x: x inputs to be used to calculate the loss function
    :param y: y outputs that are used to calculate the loss
    :param lr: learning rate for the algorithm that adjusts the gradient so that it does not overshoot
    :param epochs: number of iterations to be run
    :return: the new weights and history of the loss values
    '''
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
