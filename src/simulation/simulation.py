#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:06:58 2022

@author: magicbycalvin
"""

import logging
from threading import Thread, Event
import time

import numpy as np


class DubinsCar(Thread):
    np = np
    def __init__(self, trajectory, update_freq=100):
        super().__init__()
        self._stop_event = Event()

        self.trajectory_history = [trajectory]
        self._trajectory = trajectory
        self._traj_buffer = []

        self.update_freq = update_freq
        self.update_period = 1/update_freq

        self.t0 = None
        self.tf = None

        x = self.trajectory(0.0)
        xdot = self.trajectory.diff()(0.0)
        # xddot = self.trajectory.dif().diff()(0.0)
        self.state = np.array([x[0], x[1], DubinsCar.np.arctan2(xdot[1], xdot[0])])

    @property
    def trajectory(self):
        if len(self._traj_buffer) > 0 and self.get_time() >= self._traj_buffer[-1].t0:
            self._trajectory = self._traj_buffer.pop()
        return self._trajectory

    @trajectory.setter
    def trajectory(self, trajectory):
        self.trajectory_history.append(trajectory)
        self._traj_buffer.insert(0, trajectory)

    def get_time(self):
        """
        Get the vehicle's current time.

        Note that this function does not include t0 which means it might be very large. To get the vehicle's time since
        starting, use vehicle.get_time() - vehicle.t0.

        Override this function to use a different time source (e.g., grabbing ROS time instead of the CPU time)

        Returns
        -------
        float
            Vehicle's internal time without accounting for t0.

        """
        if self._stop_event.is_set():
            return self.tf
        else:
            return time.time()

    def stop(self):
        self.tf = self.get_time()
        self._stop_event.set()

    def run(self):
        t_last = 0.0
        self.t0 = self.get_time()
        while not self._stop_event.is_set():
            t = self.get_time() - self.t0

            # Update the state at a known frequency (it is possible that the frequency is LOWER than the desired)
            if t - t_last < self.update_period:
                continue

            x = self.trajectory(t)
            xdot = self.trajectory.diff()(t)
            # xddot = self.trajectory.dif().diff()(t)
            self.state = np.array([x[0], x[1], DubinsCar.np.arctan2(xdot[1], xdot[0])])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from polynomial.bernstein import Bernstein

    t0 = 0
    tf = 15
    cpts = np.array([[0, 1, 2, 3, 4, 5],
                     [4, 6, 2, 8, 7, 5]], dtype=float)
    c = Bernstein(cpts, t0, tf)
    _, c2 = c.split(7)
    c2.cpts[:, -1] = np.array([5, 0])

    veh = DubinsCar(c)

    plt.close('all')
    fig, ax = plt.subplots()
    plt.xlim([-1, 6])
    plt.ylim([-1, 9])
    veh.trajectory.plot(ax, showCpts=False)
    pos_handle = ax.plot(cpts[0, 0], cpts[1, 0], 'r^', ms=15)[0]

    new_traj = False
    veh.start()
    tstart = veh.get_time()
    t = 0
    while t < tf:
        t = veh.get_time() - tstart

        if not new_traj and t >= 5:
            c2.plot(ax, showCpts=False)
            veh.trajectory = c2
            new_traj = True

        state = veh.state
        if state is not None:
            pos_handle.set_xdata(state[0])
            pos_handle.set_ydata(state[1])

        plt.pause(1e-3)

    veh.trajectory.plot(ax, showCpts=False)
    veh.stop()

    plt.show()
