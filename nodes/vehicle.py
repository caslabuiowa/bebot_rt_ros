#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 13:58:25 2022

@author: magicbycalvin
"""
import numpy as np

from geometry_msgs.msg import PoseStamped
import rospy
from tf.transformations import quaternion_from_euler

from bebot_rt_ros.msg import BernsteinTrajectory
from polynomial.bernstein import Bernstein


class Vehicle:
    def __init__(self):
        super().__init__()

        rospy.init_node('vehicle', anonymous=True)
        self.traj_sub = rospy.Subscriber('trajectory', BernsteinTrajectory, callback=self.traj_cb, queue_size=10)
        self.pose_pub = rospy.Publisher('pose', PoseStamped, queue_size=10)

        self.rate = rospy.Rate(60)

        self.trajectory = None
        self.traj_buffer = []

    def start(self):
        while not rospy.is_shutdown():
            if self.trajectory is None:
                continue

            t = rospy.get_time() - self.t0
            if t > self.trajectory.tf:
                t = self.trajectory.tf
                rospy.logdebug('t is greater than trajectory\'s final time')

            position = self.trajectory(t)
            heading = np.arctan2(self.trajectory.diff().y(t), self.trajectory.diff().x(t))
            quat = quaternion_from_euler(0, 0, heading)

            msg = PoseStamped()
            msg.pose.position.x = position[0]
            msg.pose.position.y = position[1]
            msg.pose.orientation.x = quat[0]
            msg.pose.orientation.y = quat[1]
            msg.pose.orientation.z = quat[2]
            msg.pose.orientation.w = quat[3]
            msg.header.stamp = rospy.get_rostime()
            msg.header.frame_id = 'world'

            self.pose_pub.publish(msg)

    def traj_cb(self, data):
        cpts = []
        for pt in data.cpts:
            cpts.append([pt.x, pt.y])
        cpts = np.array(cpts, dtype=float).T
        traj = Bernstein(cpts, data.t0, data.tf)
        self.traj_buffer.append(traj)
        # TODO: Add the header's timestamp to the traj buffer to know when to start the next trajectory
        self.trajectory = traj
        self.t0 = rospy.get_time()
        rospy.loginfo(f'New trajectory recieved: {traj}')


if __name__ == '__main__':
    veh = Vehicle()
    veh.start()
