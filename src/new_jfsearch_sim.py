#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:26:48 2022

@author: magicbycalvin
"""

import logging
from queue import Queue
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from scipy.stats.qmc import Sobol

from polynomial.bernstein import Bernstein

from stoch_opt.constraint_functions import CollisionAvoidance, MaximumSpeed, MaximumAngularRate, SafeSphere
from stoch_opt.cost_functions import SumOfDistance
from stoch_opt.utils import state2cpts
from utils import setRCParams, resetRCParams
from vehicles import DubinsCarNC


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
                     obstacles=[], safe_dist=1, vmax=1, wmax=np.pi/4, rsafe=3,
                     n=7, ndim=2, nchildren=8, std=1.0, rng_seed=None, t_max=1.0, max_recursion=5):
    seed = rng_seed
    recursion_idx = 0
    trajs = []
    best_cost = np.inf
    best_traj = initial_guess

    t0 = initial_guess.t0
    tf = initial_guess.tf

    Q = Queue()
    Q.put(initial_guess)
    tstart = time.time()
    while time.time() - tstart < t_max:
        qsize = Q.qsize()
        logging.debug(f'{qsize=}')

        # Pop the current trajectory and compute it's feasibility and cost
        traj = Q.get(timeout=3)
        if is_feasible(traj, obstacles, safe_dist, vmax, wmax, rsafe):
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


def is_feasible(traj, obstacles, safe_dist, vmax, wmax, rsafe):
    constraints = [
        CollisionAvoidance(safe_dist, obstacles, elev=1000),
        MaximumSpeed(vmax),
        MaximumAngularRate(wmax),
        SafeSphere(traj.cpts[:, 0].squeeze(), rsafe)
        ]

    for cons in constraints:
        if not cons.call([traj]):
            return False

    return True


def cost_fn(traj, goal):
    sod = SumOfDistance(goal)

    return sod.call([traj])


def project_goal(x, rsafe, goal):
    x = x.squeeze()
    goal = goal.squeeze()
    u = np.linalg.norm(goal - x)
    scale = rsafe / u

    if scale < 1:
        new_goal = (goal - x)*scale + x
    else:
        new_goal = goal

    return new_goal


def initial_guess(initial_position, initial_velocity, initial_acceleration, t0, tf, n, ndim, goal):
    cpts_guess = np.empty((ndim, n+1))
    cpts_guess[:, :3] = state2cpts(initial_position, initial_velocity, initial_acceleration, t0, tf, n)
    cpts_guess[0, 3:] = np.linspace(cpts_guess[0, 2], goal[0], num=n-1)[1:]
    cpts_guess[1, 3:] = np.linspace(cpts_guess[1, 2], goal[1], num=n-1)[1:]
    traj_guess = Bernstein(cpts_guess, t0, tf)

    return traj_guess


if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(logging.CRITICAL)

    dt_sim = 1e-3
    dt = 1.0

    seed = 3
    std = 1.0
    nchildren = 8
    safe_planning_radius = 10
    n = 7
    ndim = 2
    t_max = 0.95
    t0 = 0
    tf = 10
    safe_dist = 2
    vmax = 3
    wmax = np.pi/4

    obstacles = [np.array([8, 0], dtype=float),  # Obstacle positions (m)
                 np.array([20, 2], dtype=float),
                 np.array([60, 1], dtype=float),
                 np.array([40, 2], dtype=float),
                 np.array([50, -3], dtype=float),
                 np.array([80, -3], dtype=float),
                 np.array([30, -1], dtype=float)]

    goal = np.array([100, 0], dtype=float)
    initial_position = np.array([-3, -1], dtype=float)
    initial_velocity = np.array([0.1, 0], dtype=float)
    initial_acceleration = np.array([0, 0], dtype=float)

    # cpts_guess = np.empty((ndim, n+1))
    # cpts_guess[:, :3] = state2cpts(initial_position, initial_velocity, initial_acceleration, t0, tf, n)
    # cpts_guess[0, 3:] = np.linspace(cpts_guess[0, 2], goal[0], num=n-1)[1:]
    # cpts_guess[1, 3:] = np.linspace(cpts_guess[1, 2], goal[1], num=n-1)[1:]
    # traj_guess = Bernstein(cpts_guess, t0, tf)
    traj_guess = initial_guess(initial_position, initial_velocity, initial_acceleration, t0, tf, n, ndim, goal)

    cur_goal = project_goal(initial_position, safe_planning_radius, goal)
    best_traj, best_cost, trajs = jellyfish_search(traj_guess, cur_goal,
                                                   obstacles=obstacles,
                                                   safe_dist=safe_dist,
                                                   vmax=vmax,
                                                   wmax=wmax,
                                                   rsafe=safe_planning_radius,
                                                   n=n,
                                                   ndim=ndim,
                                                   nchildren=nchildren,
                                                   std=std,
                                                   rng_seed=seed,
                                                   t_max=t_max)

    plt.close('all')
    setRCParams()
    fig, ax = plt.subplots()
    veh_pos_handle = ax.plot(initial_position[0], initial_position[1], 'r^', ms=15, zorder=3)[0]
    ax.plot(goal[0], goal[1], marker=(5, 1), color='green', ms=30)
    for obs in obstacles:
        ax.add_patch(plt.Circle(obs, radius=safe_dist, zorder=2))

    t = 0.0
    t_last = 0.0
    t_last_sim = 0.0
    vehicle = DubinsCarNC(initial_position, t0=0.0)
    t0_sim = time.time()
    cur_goal = project_goal(initial_position, safe_planning_radius, goal)
    ax.add_patch(plt.Circle(initial_position, radius=safe_planning_radius, fill=False, linestyle='--'))
    plt.plot(cur_goal[0], cur_goal[1], 'b*', ms=15)

    vehicle.trajectory = best_traj
    cur_pos = initial_position
    while np.linalg.norm(cur_pos.squeeze() - goal) > 1: #t < sim_time:
        t = time.time() - t0_sim
        print(f'{t=}')

        # Wait until it is time to continue running to simulate latency from sensors and control loops
        if t - t_last_sim < dt_sim:
            continue
        t_last_sim = t

        cur_pos, cur_vel, cur_acc = vehicle.update(t)
        if np.linalg.norm(cur_pos) > 1e3:
            print(f'[!] Current position is infeasible, {cur_pos=}')
            break
        veh_pos_handle.set_xdata(cur_pos[0])
        veh_pos_handle.set_ydata(cur_pos[1])

        if t - t_last > dt:
            x, xdot, xddot = vehicle.update(t+dt)
            cur_goal = project_goal(cur_pos, safe_planning_radius, goal)
            ax.add_patch(plt.Circle(cur_pos, radius=safe_planning_radius, fill=False, linestyle='--'))
            plt.plot(cur_goal[0], cur_goal[1], 'b*', ms=15)
            traj_guess = initial_guess(cur_pos, cur_vel, cur_acc, t, t+tf, n, cur_goal)
            best_traj, best_cost, trajs = jellyfish_search(traj_guess, cur_goal,
                                                           obstacles=obstacles,
                                                           safe_dist=safe_dist,
                                                           vmax=vmax,
                                                           wmax=wmax,
                                                           rsafe=safe_planning_radius,
                                                           n=n,
                                                           ndim=ndim,
                                                           nchildren=nchildren,
                                                           std=std,
                                                           rng_seed=seed,
                                                           t_max=t_max)
            print(f'Current Goal: {cur_goal}')
            print(f'{best_traj=}')
            # while qbest_cost == np.inf:
            #     qbest, qbest_cost, feasible_trajs, infeasible_trajs = traj_gen_proxy(t, x, xdot, xddot, cur_pos,
            #                                                                          cur_goal)
            if best_cost < 1e6:
                vehicle.trajectory = best_traj
                best_traj.plot(ax, showCpts=False)
                t_last = t

        plt.pause(1e-6)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    resetRCParams()
