import numpy as np
import model as mp

if __name__ == '__main__':
    x = 2 * np.random.rand(100, 1)
    y = 4 + 3 * x + np.random.randn(100, 1)
    theta = np.random.randn(2, 1)
    x_b = np.c_[np.ones((len(x), 1)), x]
    theta, c_hist, t_hist = mp.gradient_descent(x_b, y, theta)
    print(theta[0][0])
    print(theta[1][0])


