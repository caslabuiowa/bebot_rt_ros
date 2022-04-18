#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:26:48 2022

@author: magicbycalvin
"""

import logging
from queue import Queue
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from scipy.special import binom
from scipy.stats.qmc import Sobol

from polynomial.bernstein import Bernstein

from stoch_opt.constraint_functions import CollisionAvoidance, MaximumSpeed, MaximumAngularRate, SafeSphere
from stoch_opt.cost_functions import SumOfDistance
from stoch_opt.utils import state2cpts


def sobol_perturb(traj, goal, n=7, ndim=2, nchildren=8, std=1.0, rng_seed=None, recursion_idx=0):
    # TODO: ROTATION MATRIX! Keep X pointed at the goal, all our X values will need to be increasing while all our
    # Y (and Z for 3D) should stay near each other
    # TODO: Figure out why the Y points are biased to be lower
    m = int(np.log(nchildren)/np.log(2)) + int(np.log(n-3)/np.log(2))
    logging.debug(f'{m=}')

    cpts = traj.cpts.copy()
    x = cpts[:, 0]
    xbar = cpts[:, 3:-1].mean(axis=1)
    # print(f'{xbar=}')

    rng = default_rng(seed=rng_seed)
    x2goal = goal - x
    # Multiply the magnitude by a scaling factor so that we can explore a little bit outside of the space
    x2goal_mag = np.linalg.norm(x2goal)*1.1
    logging.debug(f'{x2goal=}')
    logging.debug(f'{xbar=}')

    # Sobol exploration for the intermediate points
    qmc_sobol = Sobol(ndim, seed=rng_seed)
    qmc_values = qmc_sobol.random_base2(m).T
    qmc_values.sort()
    logging.debug(f'{qmc_values=}')

    # Reshape the Sobol exploration
    values = []
    for i, val in enumerate(qmc_values):
        if i == 0:
            sample_pts = val.reshape((-1, nchildren))*x2goal_mag + x[i]
        else:
            sample_pts = (val.reshape((nchildren, -1)) - 0.5)*x2goal_mag/(2**recursion_idx) + xbar[i]

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

    logging.debug('-----------')

    return cpts_list


def jellyfish_search(initial_guess, goal,
                     obstacles=[], safe_dist=1, vmax=1, wmax=np.pi/4,
                     n=7, ndim=2, nchildren=8, std=1.0, rng_seed=None, t_max=1.0, max_recursion=5):
    seed = rng_seed
    recursion_idx = 0
    trajs = []
    best_cost = np.inf
    best_traj = initial_guess

    Q = Queue()
    Q.put(traj_guess)
    tstart = time.time()
    while time.time() - tstart < t_max:
        qsize = Q.qsize()
        logging.debug(f'{qsize=}')

        # Pop the current trajectory and compute it's feasibility and cost
        traj = Q.get(timeout=3)
        if is_feasible(traj, obstacles, safe_dist, vmax, wmax):
            cost = cost_fn(traj, goal)
            trajs.append((cost, traj))
            if cost < best_cost:
                best_traj = traj
                best_cost = cost
        else:
            trajs.append((np.inf, traj))

        # Quasi random perturbation of the trajectory
        cpts_list = sobol_perturb(traj, goal, n=n, ndim=ndim, nchildren=nchildren, std=std, rng_seed=seed,
                                  recursion_idx=recursion_idx)
        # Increment the seed so we don't get the same perturbation every time
        if seed is not None:
            seed += 1

        # Push the new trajectories onto the queue
        for cpts in cpts_list:
            new_traj = Bernstein(cpts, t0, tf)
            Q.put(new_traj)

        # Keep track of our recursion depth (even though this isn't a recursive function, we're essentially measuring
        # the depth of the tree we are searching)
        if qsize == nchildren**recursion_idx:
            recursion_idx += 1
            logging.info(f'Increasing recursion count to {recursion_idx}')
            if recursion_idx > max_recursion:
                break

    return best_traj, best_cost, trajs


def is_feasible(traj, obstacles, safe_dist, vmax, wmax):
    constraints = [
        CollisionAvoidance(safe_dist, obstacles, elev=1000),
        MaximumSpeed(vmax),
        MaximumAngularRate(wmax),
        # SafeSphere(cur_pos, rsafe)
        ]

    for cons in constraints:
        if not cons.call([traj]):
            return False

    return True


def cost_fn(traj, goal):
    sod = SumOfDistance(goal)

    return sod.call([traj])

if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(logging.INFO)

    seed = 3
    std = 1.0
    nchildren = 8
    n = 7
    ndim = 2
    t_max = 1.0
    t0 = 0
    tf = 100
    safe_dist = 2
    vmax = 30
    wmax = 1e3*np.pi/4

    obstacles = [np.array([8, 0], dtype=float),  # Obstacle positions (m)
                 np.array([20, 2], dtype=float),
                 np.array([60, 1], dtype=float),
                 np.array([40, 2], dtype=float),
                 np.array([80, -3], dtype=float),
                 np.array([30, -1], dtype=float)]

    goal = np.array([100, 0], dtype=float)
    initial_position = np.array([-3, -1], dtype=float)
    initial_velocity = np.array([0.1, 0], dtype=float)
    initial_acceleration = np.array([0, 0], dtype=float)

    cpts_guess = np.empty((ndim, n+1))
    cpts_guess[:, :3] = state2cpts(initial_position, initial_velocity, initial_acceleration, t0, tf, n)
    cpts_guess[0, 3:] = np.linspace(cpts_guess[0, 2], goal[0], num=n-1)[1:]
    cpts_guess[1, 3:] = np.linspace(cpts_guess[1, 2], goal[1], num=n-1)[1:]
    traj_guess = Bernstein(cpts_guess, t0, tf)

    best_traj, best_cost, trajs = jellyfish_search(traj_guess, goal,
                                                   obstacles=obstacles,
                                                   safe_dist=safe_dist,
                                                   vmax=vmax,
                                                   wmax=wmax,
                                                   n=n,
                                                   ndim=ndim,
                                                   nchildren=nchildren,
                                                   std=std,
                                                   rng_seed=seed,
                                                   t_max=t_max)

    plt.close('all')
    fig, ax = plt.subplots()
    [i[1].plot(ax, showCpts=False, c='g') if i[0] < np.inf else i[1].plot(ax, showCpts=False, c='r') for i in trajs]

    ax.plot(goal[0], goal[1], marker=(5, 1), color='green', ms=30)
    for obs in obstacles:
        ax.add_patch(plt.Circle(obs, radius=safe_dist, zorder=2))

    fig2, ax2 = plt.subplots()
    [i[1].plot(ax2, showCpts=True) for i in trajs[:nchildren+1]]

    feasible_trajs = [i[1] for i in trajs if i[0] < np.inf]
    infeasible_trajs = [i[1] for i in trajs if i[0] == np.inf]

    fig3, ax3 = plt.subplots()
    [i.plot(ax3, showCpts=False) for i in infeasible_trajs[40:50]]
    for obs in obstacles:
        ax3.add_patch(plt.Circle(obs, radius=safe_dist, zorder=2))

    fig4, ax4 = plt.subplots()
    bad_pts = []
    for bad in infeasible_trajs[40:50]:
        for obs in obstacles:
            cpts = np.array([[i]*(n+1) for i in obs[:ndim]], dtype=float)
            c_obs = Bernstein(cpts, t0, tf)

            dist = (bad - c_obs).normSquare()
            dist.plot(ax4, showCpts=False)
            bad_pts.append(dist.elev(30).cpts.squeeze())

    fig5, ax5 = plt.subplots()
    [i[1].plot(ax5, showCpts=False) for i in trajs]


    # x_val = []
    # y_val = []
    # for i in range(3000):
    #     sob = Sobol(2, seed=i)
    #     tmp = sob.random_base2(5).T
    #     tmp.sort()
    #     x_val.append(tmp[0, :])
    #     y_val.append(tmp[1, :])

    # print(f'{np.mean(x_val)=}')
    # print(f'{np.mean(y_val)=}')
