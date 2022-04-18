#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:58:40 2022

@author: magicbycalvin
"""

from threading import Thread, Event
import time

import numpy as np
from numpy.random import default_rng

from polynomial.bernstein import Bernstein
from stoch_opt import JellyfishSearch
from stoch_opt.constraint_functions import CollisionAvoidance, MaximumSpeed, MaximumAngularRate, SafeSphere
from stoch_opt.constraint_functions import MinimumFinalTime
from stoch_opt.cost_functions import SumOfDistance, DistanceToGoal
from stoch_opt.utils import state2cpts


class JellyfishController:
    def __init__(self, goal, obstacles, ndim, n, print_iter=False, rw_std=1.0, rng_seed=None, nchildren=2, t_max=1.0,
                 dt=1.0, p_exp=0.5, s_local_radius=15.0, d_safe=3.0, vmax=10.0, wmax=1.0, alpha=0.95):
        """


        Parameters
        ----------
        goal : numpy.ndarray
            One dimensional array representing the desired final position of the vehicle. Should have the same number
            of dimensions that the rest of the system has (e.g, 2D or 3D) in the form of [x, y, z, ...].
        obstacles : [numpy.ndarray]
            List of obstacles using the same one dimensional array scheme as the goal.
        ndim : int
            Number of dimensions of the system. Very unlikely that it will be any value other than 2 or 3, although
            higher or lower dimensions are allowed if the rest of your cost and constraint functions can handle it.
        n : int
            Degree of the Bernstein polynomials being used. This is one less than the number of control points.
        print_iter : bool, optional
            If True, print out each iteration of the optimizer. The default is False.
        rw_std : float, optional
            Standard deviation of the random walk performed by the perturb function. The default is 1.0.
        rng_seed : int, optional
            Seed for the random number generator (RNG). The RNG used is provided by numpy. The default is None.
        nchildren : int, optional
            Number of children for the Jellyfish search. Must be at least 1. Lower numbers will encourage more
            exploration while higher numbers will encourage more clustering. The default is 2.
        t_max : float, optional
            Maximum computation time for the Jellyfish search. The default is 1.0.
        dt : float, optional
            Timestep between trajectory plans.
        p_exp : float, optional
            Probability of exploring during a Jellyfish search iteration. The default is 0.5.
        s_local_radius : float, optional
            Radius defining the safe set for the robot, e.g., the sensor range. The default is 15.0.
        d_safe : float, optional
            Minimum safe distance between the vehicle and obstacles. The default is 3.0.
        vmax : float, optional
            Maximum speed of the vehicle. The default is 10.0.
        wmax : float, optional
            Maximum angular rate of the vehicle. The default is 1.0.
        alpha : float, optional
            Hyperparameter balancing the two costs in the cost function. The default is 0.95.
            cost_fn = alpha*SumOfDistance + (1-alpha)*DistanceToGoal,
            where SumOfDistance is the running cost and DistanceToGoal is the terminal cost.

        Returns
        -------
        None.

        """
        super().__init__()
        self.goal = goal
        self.obstacles = obstacles
        self.ndim = ndim
        self.n = n
        self.print_iter = print_iter
        self.rw_std = rw_std
        self.rng_seed = rng_seed
        self.rng = default_rng(rng_seed)
        self.nchildren = nchildren
        self.t_max = t_max
        self.dt = dt
        self.p_exp = p_exp
        self.s_local_radius = s_local_radius
        self.d_safe = d_safe
        self.vmax = vmax
        self.wmax = wmax
        self.alpha = alpha

        self._running = Event()
        self.result = None

    @property
    def is_running(self):
        return self._running.is_set()

    def solve(self, t0, x0, x0dot, x0ddot, cur_pos):
        if not self._running.is_set():
            self._running.set()
            thread = Thread(target=self.generate_trajectory, args=(t0, x0, x0dot, x0ddot, cur_pos))
            thread.start()
        else:
            print('[!] Warning: Jellyfish controller is already running.')

    def generate_trajectory(self, t0, x0, x0dot, x0ddot, cur_pos):
        cpts = np.empty((self.ndim, self.n+1))
        tf = t0 + 150.0

        cpts[:, :3] = state2cpts(x0, x0dot, x0ddot, t0, tf, self.n)
        cpts[:, 3:] = np.vstack([np.linspace(cpts[0, 2], self.goal[0], self.n-1)[1:],
                                 np.linspace(cpts[1, 2], self.goal[1], self.n-1)[1:]])

        # Create a Bernstein polynomial of the initial trajectory
        c = Bernstein(cpts, t0=t0, tf=tf)
        assert np.allclose(c.cpts[:, 0], x0.squeeze()), 'Initial position incorrect.'
        assert np.allclose(c.diff().cpts[:, 0], x0dot.squeeze()), 'Initial velocity incorrect.'
        assert np.allclose(c.diff().diff().cpts[:, 0], x0ddot.squeeze()), 'Initial acceleration incorrect.'

        cost_fn = self.alpha*SumOfDistance(self.goal) + (1-self.alpha)*DistanceToGoal(self.goal)
        constraints = [
            CollisionAvoidance(self.d_safe, self.obstacles),
            MaximumSpeed(self.vmax),
            MaximumAngularRate(self.wmax),
            MinimumFinalTime(self.dt + 0.5),
            SafeSphere(cur_pos, self.s_local_radius)
            ]

        jelly = JellyfishSearch(cost_fn,
                                [c],
                                constraints=constraints,
                                random_walk_std=self.rw_std,
                                rng_seed=self.rng_seed,
                                nchildren=self.nchildren,
                                max_time=self.t_max,
                                exploration_prob=self.p_exp)
        if self.rng_seed is not None:
            self.rng_seed += 1  # Increment the seed each time so that we don't get the same trajectory every time
        qbest, qbest_cost = jelly.solve(print_iter=self.print_iter)

        feasible_trajs = jelly.feasible_trajs.copy()
        infeasible_trajs = jelly.infeasible_trajs.copy()

        self.result = (qbest, qbest_cost, feasible_trajs, infeasible_trajs)
        self._running.clear()


if __name__ == '__main__':
    pass
