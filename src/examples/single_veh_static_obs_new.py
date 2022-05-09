#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 11:36:52 2022

@author: magicbycalvin
"""

# import time

import matplotlib.pyplot as plt
import numpy as np

from control import JellyfishController
# from polynomial.bernstein import Bernstein
from simulation import DubinsCar
# from stoch_opt import JellyfishSearch
# from stoch_opt.constraint_functions import CollisionAvoidance, MaximumSpeed, MaximumAngularRate, SafeSphere
# from stoch_opt.constraint_functions import MinimumFinalTime
# from stoch_opt.cost_functions import SumOfDistance, DistanceToGoal
# from stoch_opt.utils import state2cpts
from utils import setRCParams, resetRCParams
# from vehicles import DubinsCarNC


# def generate_trajectory(t0, x0, x0dot, x0ddot, obstacles, ndim, n, cur_pos, goal, dt,
#                         rng_seed=None, rw_std=1.0, nchildren=2, Tmax=1.0, print_iter=False, rsafe=3,
#                         vmax=10.0, wmax=1.0, alpha=0.95):
#     cpts = np.empty((ndim, n+1))
#     tf = t0 + 15.0

#     cpts[:, :3] = state2cpts(x0, x0dot, x0ddot, t0, tf, n)
#     cpts[:, 3:] = np.vstack([np.linspace(cpts[0, 2], goal[0], n-1)[1:],
#                              np.linspace(cpts[1, 2], goal[1], n-1)[1:]])

#     # Create a Bernstein polynomial of the initial trajectory
#     c = Bernstein(cpts, t0=t0, tf=tf)
#     assert np.allclose(c.cpts[:, 0], x0.squeeze()), 'Initial position incorrect.'
#     assert np.allclose(c.diff().cpts[:, 0], x0dot.squeeze()), 'Initial velocity incorrect.'
#     assert np.allclose(c.diff().diff().cpts[:, 0], x0ddot.squeeze()), 'Initial acceleration incorrect.'

#     cost_fn = alpha*SumOfDistance(goal) + (1-alpha)*DistanceToGoal(goal)
#     constraints = [
#         CollisionAvoidance(safe_dist, obstacles),
#         MaximumSpeed(vmax),#, elev=50),
#         MaximumAngularRate(maximum_angular_rate),#, elev=100),
#         MinimumFinalTime(dt + 0.5),
#         SafeSphere(cur_pos, rsafe)
#         ]

#     jelly = JellyfishSearch(cost_fn,
#                             [c],
#                             constraints=constraints,
#                             random_walk_std=rw_std,
#                             rng_seed=rng_seed,
#                             nchildren=nchildren,
#                             max_time=Tmax,
#                             exploration_prob=exploration_prob)
#     qbest, qbest_cost = jelly.solve(print_iter=print_iter)

#     feasible_trajs = jelly.feasible_trajs.copy()
#     infeasible_trajs = jelly.infeasible_trajs.copy()

#     return qbest, qbest_cost, feasible_trajs, infeasible_trajs


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


if __name__ == '__main__':
    # Mission Parameters
    rng_seed = 3
    std = 1.5
    exploration_prob = 0.3
    nchildren = 2
    ndim = 2  # This example only works for 2D
    n = 10
    dt = 1.0  # Timestep between plans (s)
    sim_time = 15  # Amount of time to run the simulation (s)
    dt_sim = 1e-3  # Simulation timestep (s)
    safe_dist = 2  # Minimum safe distance between the vehicle and obstacles (m)
    maximum_speed = 1  # m/s
    maximum_angular_rate = np.pi/4  # rad/s
    safe_planning_radius = 10  # Radius that defines the safe area to generate the local plan (m)
    Tmax = 0.95  # Maximum computation time for the jellyfish planner (s)

    # Obstacle and goal positions
    obstacles = [np.array([8, 4], dtype=float),  # Obstacle positions (m)
                 np.array([20, 2], dtype=float),
                 np.array([60, 5], dtype=float),
                 np.array([40, 2], dtype=float),
                 np.array([80, 9], dtype=float),
                 np.array([30, 8], dtype=float)]
    goal = np.array([100, 10], dtype=float)  # Goal position (m)

    # Initial state of vehicle
    x0 = np.array([0, 0], dtype=float)
    x0dot = np.array([1, 0], dtype=float)
    x0ddot = np.array([0, 0.1], dtype=float)

    # def traj_gen_proxy(t, x, xdot, xddot, cur_pos, goal):
    #     return generate_trajectory(t, x, xdot, xddot, obstacles, ndim, n, cur_pos, goal, dt,
    #                                rng_seed=rng_seed, rw_std=std, nchildren=nchildren, Tmax=Tmax, print_iter=False,
    #                                rsafe=safe_planning_radius, vmax=maximum_speed, wmax=maximum_angular_rate)

    # Set up plotting
    plt.close('all')
    setRCParams()
    fig, ax = plt.subplots()
    ax.plot(goal[0], goal[1], marker=(5, 1), color='green', ms=30)
    for obs in obstacles:
        ax.add_patch(plt.Circle(obs, radius=safe_dist, zorder=2))

    # # Initialize current (projected) goal and plot
    # cur_goal = project_goal(x0, safe_planning_radius, goal)
    # ax.add_patch(plt.Circle(x0, radius=safe_planning_radius, fill=False, linestyle='--'))
    # plt.plot(cur_goal[0], cur_goal[1], 'b*', ms=15)

    # Generate the first trajectory and initialize the vehicle
    # qbest, qbest_cost, feasible_trajs, infeasible_trajs = traj_gen_proxy(0.0, x, xdot, xddot, x, cur_goal)
    jf_planner = JellyfishController(goal, obstacles, ndim, n, t_max=10, nchildren=nchildren, rw_std=std)
    jf_planner.solve(0.0, x0, x0dot, x0ddot, x0)
    while jf_planner.is_running:
        pass
    qbest, qbest_cost, feasible_trajs, infeasible_trajs = jf_planner.result
    vehicle = DubinsCar(qbest[0])
    vehicle.trajectory.plot(ax, showCpts=False)
    jf_planner.t_max = Tmax

    try:
        # Start the simulation
        vehicle.start()
        veh_pos_handle = ax.plot(vehicle.state[0], vehicle.state[1], 'r^', ms=15, zorder=3)[0]
        new_traj = False
        while vehicle.t0 is None: pass  # Make sure the vehicle's internal simulation has started

        x = vehicle.trajectory(dt)
        xdot = vehicle.trajectory.diff()(dt)
        xddot = vehicle.trajectory.diff().diff()(dt)

        # Generate the trajectory
        jf_planner.solve(0.0, x, xdot, xddot, x0)

        while np.linalg.norm(vehicle.state[:2].squeeze() - goal) > 0.1:
            t = vehicle.get_time() - vehicle.t0
            print(f't: {t:0.3f}')

            cur_pos = vehicle.state
            veh_pos_handle.set_xdata(cur_pos[0])
            veh_pos_handle.set_ydata(cur_pos[1])

            if not jf_planner.is_running:
                print('starting new traj')
                # cur_goal = project_goal(cur_pos[:2], safe_planning_radius, goal)
                # ax.add_patch(plt.Circle(cur_pos[:2], radius=safe_planning_radius, fill=False, linestyle='--'))
                # plt.plot(cur_goal[0], cur_goal[1], 'b*', ms=15)

                qbest, qbest_cost, feasible_trajs, infeasible_trajs = jf_planner.result

                # If the trajectory generation was successful, save it
                if qbest_cost < 1e6:
                    print('saving traj')
                    vehicle.trajectory = qbest[0]
                    qbest[0].plot(ax, showCpts=False)

                # Get the predicted future state
                x = vehicle.trajectory(t + dt)
                xdot = vehicle.trajectory.diff()(t + dt)
                xddot = vehicle.trajectory.diff().diff()(t + dt)

                # Generate the trajectory
                jf_planner.solve(t, x, xdot, xddot, cur_pos)
                # print(f'Current Goal: {cur_goal}')

            # if new_traj and not jf_planner.is_running:
            #     print('getting new traj')
            #     new_traj = False
            #     qbest, qbest_cost, feasible_trajs, infeasible_trajs = jf_planner.result
            #     print(f'cost: {qbest_cost}')

                # # If the trajectory generation was successful, save it
                # if qbest_cost < 1e6:
                #     print('saving new traj')
                #     vehicle.trajectory = qbest[0]
                #     qbest[0].plot(ax, showCpts=False)

            plt.pause(1e-6)
    finally:
        vehicle.stop()

    # Plot the global trajectory
    trajs = vehicle.trajectory_history[1:]
    global_traj = []
    for i in range(len(trajs)-1):
        left, right = trajs[i].split(trajs[i+1].t0+dt)
        global_traj.append(left)
    global_traj.append(trajs[-1])

    fig2, ax2 = plt.subplots()
    [i.plot(ax2, showCpts=False) for i in global_traj]
    ax2.plot(goal[0], goal[1], marker=(5, 1), color='green', ms=30)
    for obs in obstacles:
        ax2.add_patch(plt.Circle(obs, radius=safe_dist, zorder=2))

    plt.show()
    resetRCParams()
