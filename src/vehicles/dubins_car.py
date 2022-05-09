#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 09:43:45 2022

@author: magicbycalvin
"""

import numpy as np
from scipy.integrate import solve_ivp

from .base_vehicle import BaseVehicleNC


class DubinsCar:
    def __init__(self, x0=0.0, y0=0.0, theta0=0.0, t0=0.0, gps_std=0.0, gps_heading_std=0.0, rng_seed=None):
        self._pose = np.array([x0, y0, theta0], dtype=np.float64)
        self._t = np.float64(t0)

        self._v = 0.0
        self._w = 0.0

        self._cov = np.diag([gps_std**2]*2 + [gps_heading_std**2])
        self._rng = np.random.default_rng(rng_seed)

        self._hist = [(self._t, self._pose[0], self._pose[1], self._pose[2], self._v, self._w)]
        self._gps_hist = []

    @property
    def t(self):
        return self._t

    @property
    def pose(self):
        return self._pose.flatten()

    @property
    def x(self):
        return self._pose[0]

    @property
    def y(self):
        return self._pose[1]

    @property
    def theta(self):
        return self._pose[2]

    @property
    def v(self):
        return self._v

    @property
    def w(self):
        return self._w

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, val):
        assert np.array_equal(val, val.T), 'The covariance matrix must be symmetric.'
        assert np.all(np.linalg.eigvals(val) >= 0), 'The covariance matrix must be positive definite.'

        self._cov = val

    @property
    def history(self):
        return np.array(self._hist)

    def get_gps(self, return_nominal=False):
        measured = self._rng.multivariate_normal(self._pose.squeeze(), self._cov)
        self._gps_hist.append((self._t, measured))

        if return_nominal:
            nominal = self._pose
            return measured, nominal
        else:
            return measured

    def manual_update(self, t, x, y, theta, v=0.0, w=0.0):
        self._t = t
        self._pose = np.array([x, y, theta], dtype=np.float64)
        self._v = v
        self._w = w
        self._hist.append([self._t, self._pose[0], self._pose[1], self._pose[2], self._v, self._w])

    def update(self, t):
        assert t > self.t, f'The current update time, {t}, must be larger than the last update time, {self.t}.'

        sol = solve_ivp(self._model, [self.t, t], self.pose)

        self._t = t
        self._pose = sol.y[:, -1]
        self._hist.append((self._t, self._pose[0], self._pose[1], self._pose[2], self._v, self._w))

    def command(self, v, w):
        self._v = np.float64(v)
        self._w = np.float64(w)

    def _model(self, t, x):
        # xd = v*cos(theta)
        # yd = v*sin(theta)
        # thetad = w
        # x = [x, y, theta]
        y = np.empty(3)
        y[0] = self._v*np.cos(x[2])
        y[1] = self._v*np.sin(x[2])
        y[2] = self._w

        return y


class DubinsCarNC(BaseVehicleNC):
    def __init__(self, x0, t0=0.0, trajectory=None):
        super().__init__(x0, t0=t0, trajectory=trajectory)

    def update(self, t):
        if t < self.trajectory.t0:
            print('[!] t is less than t0')
        elif t > self.trajectory.tf:
            print('[!] t is greater than tf, clipping to tf')
            t = self.trajectory.tf
        x = self.trajectory(t)
        xdot = self.trajectory.diff()(t)
        xddot = self.trajectory.diff().diff()(t)

        self._pos_history.append((t, x))

        return x, xdot, xddot


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.close('all')

    x0 = 1
    y0 = 0
    theta0 = np.pi/4
    t0 = 3
    tf = 11
    veh = DubinsCar(x0, y0, theta0, t0)

    veh.command(1, 0)

    new_cmd = False
    for t in np.linspace(t0, tf, 1001)[1:]:
        if not new_cmd and t > 10:
            veh.command(2, -2*np.pi/3)
            new_cmd = True
        veh.update(t)

    poses = veh.history[:, 1:3]
    plt.figure()
    plt.plot(poses[:, 0], poses[:, 1])
