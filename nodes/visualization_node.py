#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 21:24:11 2022

@author: magicbycalvin
"""

import numpy as np

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import rospy

from bebot_rt_ros.msg import BernsteinTrajectory
from polynomial.bernstein import Bernstein


class Visualization:
    def __init__(self):
        rospy.init_node('visualizer', anonymous=True)

        self.traj_sub = rospy.Subscriber('trajectory', BernsteinTrajectory, self.traj_cb, queue_size=10)
        self.path_pub = rospy.Publisher('traj_path', Path, queue_size=10)

    def traj_cb(self, data):
        cpts = []
        for pt in data.cpts:
            cpts.append([pt.x, pt.y])
        cpts = np.array(cpts, dtype=float).T
        traj = Bernstein(cpts, data.t0, data.tf)

        path = Path()
        poses = []
        for pt in traj(np.linspace(data.t0, data.tf, 101)).T:
            tmp = PoseStamped()
            tmp.header.frame_id = 'world'
            tmp.pose.position.x = pt[0]
            tmp.pose.position.y = pt[1]
            poses.append(tmp)

        path.poses = poses
        path.header.frame_id = 'world'
        path.header.stamp = rospy.get_rostime()

        self.path_pub.publish(path)

    def start(self):
        rospy.spin()

if __name__ == '__main__':
    viz = Visualization()
    viz.start()
