#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:20:50 2022

@author: magicbycalvin
"""

import matplotlib.pyplot as plt
import numpy as np

from polynomial.bernstein import Bernstein
from stoch_opt import JellyfishSearch
from stoch_opt.constraint_functions import CollisionAvoidance, MaximumSpeed, MaximumAngularRate
from stoch_opt.cost_functions import SumOfDistance, DistanceToGoal
from utils import setRCParams, resetRCParams


def cpts_from_initial_state(x, xdot, xddot, t0, tf, n):
    x = x.flatten()
    xdot = xdot.flatten()
    xddot = xddot.flatten()

    ndim = x.shape[0]
    cpts = np.empty((ndim, 3))

    cpts[:, 0] = x
    cpts[:, 1] = xdot*((tf-t0) / n) + cpts[:, 0]
    cpts[:, 2] = xddot*((tf-t0) / n)*((tf-t0) / (n-1)) + 2*cpts[:, 1] - cpts[:, 0]

    return cpts


if __name__ == '__main__':
    ### Mission Parameters
    # rng stuff
    SEED = 3
    STD = 1
    exploration_prob = 0.3
    nchildren = 2
    ndim = 2
    n = 5
    t0 = 0
    tf = 10
    safe_dist = 2
    goal_eps = 0.5
    maximum_speed = 3  # m/s
    maximum_angular_rate = np.pi/4  # rad/s
    Tmax = 5  # s

    obs = np.array([8, 4], dtype=float)     # Obstacle position (m)
    goal = np.array([10, 10], dtype=float)  # Goal position (m)

    # Initial state of vehicle
    x = np.array([0, 0], dtype=float)
    xdot = np.array([1, 0], dtype=float)
    xddot = np.array([0, 0.1], dtype=float)
    cpts = np.empty((ndim, n+1))

    cpts[:, :3] = cpts_from_initial_state(x, xdot, xddot, t0, tf, n)
    cpts[:, 3:] = np.vstack([np.linspace(cpts[0, 2], goal[0], n-1)[1:],
                             np.linspace(cpts[1, 2], goal[0], n-1)[1:]])

    # Create a Bernstein polynomial of the initial trajectory
    c = Bernstein(cpts, t0=t0, tf=tf)
    assert np.array_equal(c.cpts[:, 0], x), 'Initial position incorrect.'
    assert np.array_equal(c.diff().cpts[:, 0], xdot), 'Initial velocity incorrect.'
    assert np.array_equal(c.diff().diff().cpts[:, 0], xddot), 'Initial acceleration incorrect.'

    alpha = 0.5
    cost_fn = (1-alpha)*SumOfDistance(goal) + alpha*DistanceToGoal(goal)
    constraints = [
        CollisionAvoidance(safe_dist, [obs]),
        MaximumSpeed(maximum_speed),
        MaximumAngularRate(maximum_angular_rate)
        ]

    jelly = JellyfishSearch(cost_fn,
                            [c],
                            constraints=constraints,
                            random_walk_std=STD,
                            rng_seed=SEED,
                            nchildren=nchildren,
                            max_time=Tmax,
                            exploration_prob=exploration_prob)
    qbest, qbest_cost = jelly.solve(print_iter=True)

    feasible_trajs = jelly.feasible_trajs.copy()
    infeasible_trajs = jelly.infeasible_trajs.copy()

    ### Plot the results
    plt.close('all')
    setRCParams()
    fig, ax = plt.subplots()

    if len(infeasible_trajs) > 0:
        infeasible_trajs[0][0][0].plot(ax, showCpts=False, color='red', alpha=0.5, label='infeasible')
        for qbad in infeasible_trajs[1:]:
            for traj in qbad[0]:
                traj.plot(ax, showCpts=False, color='red', alpha=0.3, label=None)

    if len(feasible_trajs) > 0:
        feasible_trajs[0][0][0].plot(ax, showCpts=False, color='green', alpha=0.5, label='feasible')
        for qgood in feasible_trajs[1:]:
            for traj in qgood[0]:
                traj.plot(ax, showCpts=False, color='green', alpha=0.3, label=None)

    for traj in qbest:
        traj.plot(ax, showCpts=False, color='blue', lw=5, label='best')

    ax.plot(goal[0], goal[1], marker=(5, 1), color='purple', ms=50)
    ax.add_patch(plt.Circle(obs, radius=safe_dist, zorder=2))

    plt.legend(loc='upper left')

    # # Debugging stuff here
    # fig2, ax2 = plt.subplots()
    # tmp = jelly.rng.choice(list(zip(*infeasible_trajs))[0], size=30, replace=False)
    # for traj in tmp:
    #     traj[0].plot(ax2, showCpts=False)

    # ax2.plot(goal[0], goal[1], marker=(5, 1), color='purple', ms=50)
    # ax2.add_patch(plt.Circle(obs, radius=safe_dist, zorder=2))

    plt.show()
    resetRCParams()
