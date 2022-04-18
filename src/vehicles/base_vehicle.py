#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:13:29 2022

@author: magicbycalvin
"""

from abc import ABC, abstractmethod


class BaseVehicle(ABC):
    def __init__(self, x0, t0=0.0):
        self.x0 = x0
        self.t0 = t0

        self.t = t0
        self.state = x0
        self._history = [(t0, x0)]

    @abstractmethod
    def update(self, t):
        pass


class BaseVehicleNC(ABC):
    def __init__(self, x0, t0=0.0, trajectory=None):
        self.x0 = x0
        self.t0 = t0
        self._trajectory = trajectory

        self.t = t0
        self.state = x0
        self._pos_history = [(t0, x0)]
        self._traj_history = [trajectory]

    @property
    def trajectory(self):
        return self._trajectory

    @trajectory.setter
    def trajectory(self, new_traj):
        self._traj_history.append(new_traj)

        self._trajectory = new_traj

    @abstractmethod
    def update(self, t):
        pass
