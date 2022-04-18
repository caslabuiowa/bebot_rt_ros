#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:31:34 2022

@author: magicbycalvin
"""
import os

import numpy as np


def save_trajs(trajs, fname=None):
    if fname is None:
        fname = 'trajectories'

    name_idx = 0
    fname_new = fname
    while os.path.isfile(fname_new):
        fname_new = fname + f'{name_idx}'
        name_idx += 1

    save_list = []
    for traj_set in trajs:
        for traj in traj_set:
            t0 = traj.t0
            tf = traj.tf
            cpts = traj.cpts

            save_list.append((cpts, t0, tf))

    np.save(fname_new, save_list)



if __name__ == '__main__':
    pass
