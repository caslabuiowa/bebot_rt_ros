#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 21:22:27 2022

@author: magicbycalvin
"""

from queue import Queue

import numpy as np
from numpy.random import default_rng
from scipy.stats.qmc import Sobol

from bebot_rt_ros.msg import BernsteinTrajectory, BernsteinTrajectoryArray
from geometry_msgs.msg import Point
import rospy

from polynomial.bernstein import Bernstein
from stoch_opt.constraint_functions import CollisionAvoidance, MaximumSpeed, MaximumAngularRate, SafeSphere
from stoch_opt.cost_functions import SumOfDistance
from stoch_opt.utils import state2cpts


def sobol_perturb(traj, goal, n=7, ndim=2, nchildren=8, std=1.0, rng_seed=None, recursion_idx=0):
    # TODO: ROTATION MATRIX! Keep X pointed at the goal, all our X values will need to be increasing while all our
    # Y (and Z for 3D) should stay near each other
    # TODO: Figure out why the Y points are biased to be lower
    m = int(np.log(nchildren)/np.log(2)) + int(np.log(n-3)/np.log(2))
    rospy.logdebug(f'{m=}')

    cpts = traj.cpts.copy()
    x = cpts[:, 0]
    xbar = cpts[:, 3:-1].mean(axis=1)
    # print(f'{xbar=}')

    rng = default_rng(seed=rng_seed)
    x2goal = goal - x
    # Multiply the magnitude by a scaling factor so that we can explore a little bit outside of the space
    x2goal_mag = np.linalg.norm(x2goal)*1.1
    rospy.logdebug(f'{x2goal=}')
    rospy.logdebug(f'{xbar=}')

    # Sobol exploration for the intermediate points
    qmc_sobol = Sobol(ndim, seed=rng_seed)
    qmc_values = qmc_sobol.random_base2(m).T
    qmc_values.sort()
    rospy.logdebug(f'{qmc_values=}')

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
            # cpts[j, -1] = rng.normal(loc=goal[j], scale=std)
            cpts[j, -1] = cpts[j, -2]  # TODO: pick a better way to get the final cpt
        cpts_list.append(cpts.copy())

    rospy.logdebug('-----------')

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
    tstart = rospy.get_time()
    while rospy.get_time() - tstart < t_max:
        qsize = Q.qsize()
        rospy.logdebug(f'{qsize=}')

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
            rospy.loginfo(f'Increasing recursion count to {recursion_idx}')
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


class JellyfishController:
    def __init__(self, obstacles, safe_dist, vmax, wmax, safe_planning_radius, n, ndim, nchildren, std, t_max,
                 rng_seed=None):
        rospy.init_node('jellyfish_node')
        self.traj_pub = rospy.Publisher('trajectory', BernsteinTrajectory, queue_size=10)
        self.traj_array_pub = rospy.Publisher('jfs_guess', BernsteinTrajectoryArray, queue_size=10)

        self.n = n
        self.ndim = ndim
        self.obstacles = obstacles
        self.safe_dist = safe_dist
        self.vmax = vmax
        self.wmax = wmax
        self.safe_planning_radius = safe_planning_radius
        self.nchildren = nchildren
        self.std = std
        self.t_max = t_max
        self.rng_seed = rng_seed

        self.rate = rospy.Rate(1)

    def start(self, initial_position, initial_velocity, initial_acceleration, goal):
        t0 = 0
        tf = 10
        dt = 1
        while not rospy.is_shutdown():
            cur_goal = project_goal(initial_position, self.safe_planning_radius, goal)
            rospy.loginfo(f'Current goal: {cur_goal}')

            traj_guess = initial_guess(initial_position, initial_velocity, initial_acceleration, t0, tf,
                                       self.n, self.ndim, cur_goal)
            best_traj, best_cost, trajs = jellyfish_search(traj_guess, cur_goal,
                                                           obstacles=self.obstacles,
                                                           safe_dist=self.safe_dist,
                                                           vmax=self.vmax,
                                                           wmax=self.wmax,
                                                           rsafe=self.safe_planning_radius,
                                                           n=self.n,
                                                           ndim=self.ndim,
                                                           nchildren=self.nchildren,
                                                           std=self.std,
                                                           rng_seed=self.rng_seed,
                                                           t_max=self.t_max)
            #
            #
            #
            initial_position = best_traj(t0 + dt)
            initial_velocity = best_traj.diff()(t0 + dt)
            initial_acceleration = best_traj.diff().diff()(t0 + dt)

            msg = BernsteinTrajectory()
            cpts = []
            for pt in best_traj.cpts.T:
                tmp = Point()
                tmp.x = pt[0]
                tmp.y = pt[1]
                cpts.append(tmp)
            msg.cpts = cpts
            msg.t0 = t0
            msg.tf = tf
            msg.header.stamp = rospy.get_rostime()
            msg.header.frame_id = 'world'
            self.traj_pub.publish(msg)

            # Publish traj here
            # Vehicle should monitor time so that it follows the new traj asap

            costs, _ = zip(*trajs)
            traj_idxs = np.argsort(costs)
            traj_list = []
            for i in traj_idxs[:10]:
                traj = trajs[i][1]
                bern_traj_msg = BernsteinTrajectory()
                cpts = []
                for pt in traj.cpts.T:
                    tmp = Point()
                    tmp.x = pt[0]
                    tmp.y = pt[1]
                    cpts.append(tmp)
                bern_traj_msg.cpts = cpts
                bern_traj_msg.t0 = t0
                bern_traj_msg.tf = tf
                # bern_traj_msg.header.stamp = rospy.get_rostime()
                bern_traj_msg.header.frame_id = 'world'
                traj_list.append(bern_traj_msg)

            traj_array_msg = BernsteinTrajectoryArray()
            traj_array_msg.trajectories = traj_list
            self.traj_array_pub.publish(traj_array_msg)

            self.rate.sleep()
            # Need to skip rate.sleep if the search doesn't find a solution in time (well, does it matter since the
            # jf search always lasts a second?)


if __name__ == '__main__':
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
    initial_position = np.array([0, 0], dtype=float)
    initial_velocity = np.array([1, 0], dtype=float)
    initial_acceleration = np.array([0.1, 1], dtype=float)

    # traj_guess = initial_guess(initial_position, initial_velocity, initial_acceleration, t0, tf, n, ndim, goal)

    # cur_goal = project_goal(initial_position, safe_planning_radius, goal)
    # best_traj, best_cost, trajs = jellyfish_search(traj_guess, cur_goal,
    #                                                obstacles=obstacles,
    #                                                safe_dist=safe_dist,
    #                                                vmax=vmax,
    #                                                wmax=wmax,
    #                                                rsafe=safe_planning_radius,
    #                                                n=n,
    #                                                ndim=ndim,
    #                                                nchildren=nchildren,
    #                                                std=std,
    #                                                rng_seed=seed,
    #                                                t_max=t_max)


    # t = 0.0
    # t_last = 0.0
    # t_last_sim = 0.0

    # cur_goal = project_goal(initial_position, safe_planning_radius, goal)
    # ax.add_patch(plt.Circle(initial_position, radius=safe_planning_radius, fill=False, linestyle='--'))
    # plt.plot(cur_goal[0], cur_goal[1], 'b*', ms=15)

    # vehicle.trajectory = best_traj
    # cur_pos = initial_position
    # while np.linalg.norm(cur_pos.squeeze() - goal) > 1: #t < sim_time:
    #     t = time.time() - t0_sim
    #     print(f'{t=}')

    #     # Wait until it is time to continue running to simulate latency from sensors and control loops
    #     if t - t_last_sim < dt_sim:
    #         continue
    #     t_last_sim = t

    #     cur_pos, cur_vel, cur_acc = vehicle.update(t)
    #     if np.linalg.norm(cur_pos) > 1e3:
    #         print(f'[!] Current position is infeasible, {cur_pos=}')
    #         break
    #     veh_pos_handle.set_xdata(cur_pos[0])
    #     veh_pos_handle.set_ydata(cur_pos[1])

    #     if t - t_last > dt:
    #         x, xdot, xddot = vehicle.update(t+dt)
    #         cur_goal = project_goal(cur_pos, safe_planning_radius, goal)
    #         ax.add_patch(plt.Circle(cur_pos, radius=safe_planning_radius, fill=False, linestyle='--'))
    #         plt.plot(cur_goal[0], cur_goal[1], 'b*', ms=15)
    #         traj_guess = initial_guess(cur_pos, cur_vel, cur_acc, t, t+tf, n, cur_goal)
    #         best_traj, best_cost, trajs = jellyfish_search(traj_guess, cur_goal,
    #                                                        obstacles=obstacles,
    #                                                        safe_dist=safe_dist,
    #                                                        vmax=vmax,
    #                                                        wmax=wmax,
    #                                                        rsafe=safe_planning_radius,
    #                                                        n=n,
    #                                                        ndim=ndim,
    #                                                        nchildren=nchildren,
    #                                                        std=std,
    #                                                        rng_seed=seed,
    #                                                        t_max=t_max)
    #         print(f'Current Goal: {cur_goal}')
    #         print(f'{best_traj=}')
    #         # while qbest_cost == np.inf:
    #         #     qbest, qbest_cost, feasible_trajs, infeasible_trajs = traj_gen_proxy(t, x, xdot, xddot, cur_pos,
    #         #                                                                          cur_goal)
    #         if best_cost < 1e6:
    #             vehicle.trajectory = best_traj
    #             best_traj.plot(ax, showCpts=False)
    #             t_last = t

    #     plt.pause(1e-6)

    jfc = JellyfishController(obstacles, safe_dist, vmax, wmax, safe_planning_radius, n, ndim, nchildren, std, t_max)
    jfc.start(initial_position, initial_velocity, initial_acceleration, goal)
