import numpy as np
import pandas as pd


def cost_function(theta, x, y):
    m = len(y)
    pred = x.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(pred - y))
    return cost


def sin_transform(x, w):
    f = np.sin(w[0] + np.dot(x.T, w[1:])).T
    return f


def square_transform(x, w):
    f = np.square(w[0] + np.dot(x.T, w[1:])).T
    return f


def gradient_descent(x, y, theta, l=0.03, iterations=100):
    m = len(y)
    c_hist = np.zeros(iterations)
    t_hist = np.zeros((iterations, 2))
    for i in range(iterations):
        pred = np.dot(x, theta)
        theta -= (1/m)*l*(x.T.dot((pred - y)))
        t_hist[i, :] = theta.T
        c_hist[i] = cost_function(theta, x, y)
    return theta, c_hist, t_hist

def gauss_newton():
    return

