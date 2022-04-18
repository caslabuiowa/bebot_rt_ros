#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:00:19 2022

@author: magicbycalvin
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from scipy.special import binom
from scipy.stats.qmc import Sobol

from polynomial.bernstein import Bernstein

from stoch_opt.utils import state2cpts


def qmc(x0, t0, tf, goal, n=6, ndim=2, nchildren=8, t_max=1.0, rng_seed=None):
    x = x0[:ndim]
    xdot = x0[ndim:2*ndim]
    xddot = x0[2*ndim:3*ndim]
    m = int(np.log(nchildren)/np.log(2))

    x2goal = goal - x

    cpts = np.empty((ndim, n+1))
    cpts[:, :3] = state2cpts(x, xdot, xddot, t0, tf, n)

    qmc_sobol = Sobol(ndim, seed=rng_seed)
    qmc_values = qmc.random_base2(m)

    values = []
    for i in range(ndim):
        sample_pts = qmc_values[i, :]
        values.append()

    return cpts


if __name__ == '__main__':
    goal = np.array([100, 10], dtype=float)
    initial_position = np.array([20, -30], dtype=float)
    initial_velocity = np.array([1, 0], dtype=float)
    initial_acceleration = np.array([0, 0.1], dtype=float)
    x0 = np.append([], [initial_position, initial_velocity, initial_acceleration])

    ### Start qmc
    # Optional input parameters
    t0 = 0
    tf = 10
    n = 7
    ndim = 2
    nchildren = 8
    t_max = 1.0
    rng_seed = None
    std = 1.0
    # Start function
    # TODO: apply rotation matrix so that the x direction faces directly towards the goal
    x = x0[:ndim]
    xdot = x0[ndim:2*ndim]
    xddot = x0[2*ndim:3*ndim]
    m = int(np.log(nchildren)/np.log(2)) + int(np.log(n-3)/np.log(2))

    rng = default_rng(seed=rng_seed)
    x2goal = goal - x
    x2goal_mag = np.linalg.norm(x2goal)

    cpts = np.empty((ndim, n+1))
    cpts[:, :3] = state2cpts(x, xdot, xddot, t0, tf, n)

    # Sobol exploration for the intermediate points
    qmc_sobol = Sobol(ndim, seed=rng_seed)
    qmc_values = qmc_sobol.random_base2(m)

    # Reshape the Sobol exploration
    # TODO: ROTATION MATRIX! Keep X pointed at the goal, all our X values will need to be increasing while all our
    # Y (and Z for 3D) should stay near each other
    values = []
    for i in range(ndim):
        if i == 0:
            sample_pts = np.sort(qmc_values[:, i]).reshape((-1, nchildren))*x2goal_mag + x[i]
        else:
            sample_pts = np.sort(qmc_values[:, i]).reshape((nchildren, -1))*x2goal_mag - x2goal_mag/2 + x[i]

        for row in sample_pts:
            rng.shuffle(row)

        if i != 0:
            sample_pts = sample_pts.T

        values.append(sample_pts)

    # Gaussian exploration for the final point
    cpts_list = []
    for i in range(nchildren):
        for j, pts in enumerate(values):
            cpts[j, 3:-1] = pts[:, i]
            cpts[j, -1] = rng.normal(loc=goal[j], scale=std)
        cpts_list.append(cpts.copy())

    ### End qmc

    fig, ax = plt.subplots()
    [Bernstein(i).plot(ax) for i in cpts_list]


# if __name__ == '__main__':
#     seed = 3
#     cpts = np.array([[0, 1, 2, 3, 4, 5],
#                      [0, 3, 3, 3, 3, 0]], dtype=float)
#     b_coeffs = np.array([binom(5, k) for k in range(6)], dtype=float)

#     c1 = Bernstein(cpts)
#     c2 = Bernstein(cpts*np.vstack([np.ones(cpts.shape[1]), b_coeffs]))

#     # qmc_sobol = Sobol(1, seed=3)
#     # x_values = qmc_sobol.random_base2(3)*10
#     # y_values = qmc_sobol.random_base2(3)*10 - 5

#     rng = default_rng(seed=seed)
#     qmc_sobol2 = Sobol(2, seed=seed)
#     values = qmc_sobol2.random_base2(6)
#     x_values = values[:, 0]*10
#     x_values.sort()
#     x_values = x_values.reshape(8, -1)
#     y_values = values[:, 1]*20 - 10
#     y_values.sort()
#     y_values = y_values.reshape(8, -1)

#     for row in x_values:
#         rng.shuffle(row)

#     for row in y_values:
#         rng.shuffle(row)

#     traj_list = []
#     for i in range(8):
#         # x_tmp = [rng.choicex]
#         cpts_tmp = np.hstack((np.zeros((2, 1)), np.array([x_values[:, i], y_values[i, :]])))
#         traj_list.append(Bernstein(cpts_tmp))

#     # x_list = [(qmc_sobol.random_base2(3)*10).sort() for i in range(8)]
#     # y_list = [(qmc_sobol.random_base2(3)*10 - 5).sort() for i in range(8)]
#     # for i in range(8):
#     #     y_values = qmc_sobol.random_base(3)*10
#     #     y_values.sort()
#     #     for j in range(8):
#     #         x_values = qmc_sobol.random_base2(3)*10 - 5


#     plt.close('all')
#     ax = c1.plot()
#     c2.plot(ax)

#     fig2, ax2 = plt.subplots()
#     [i.plot(ax2, showCpts=True) for i in traj_list]
