import numpy as np


def gen_gaussian2d(m, n, amplitude, sigma_x, sigma_y):
    x = np.linspace(-m, m, 2 * m + 1)
    y = np.linspace(-n, n, 2 * n + 1)
    x, y = np.meshgrid(x, y)
    xo = 0.0
    yo = 0.0
    theta = 0.0
    offset = 0.0
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp(- (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                                     + c*((y-yo)**2)))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    sumg = g.sum()
    if sumg != 0:
        g /= sumg

    return g
