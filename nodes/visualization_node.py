#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 21:24:11 2022

@author: magicbycalvin
"""

import matplotlib.pyplot as plt
import numpy as np

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
import rospy
from visualization_msgs.msg import Marker, MarkerArray

from bebot_rt_ros.msg import BernsteinTrajectory
from polynomial.bernstein import Bernstein


class Visualization:
    def __init__(self, obstacles=[]):
        rospy.init_node('visualizer', anonymous=True)

        self.traj_sub = rospy.Subscriber('trajectory', BernsteinTrajectory, self.traj_cb, queue_size=10)
        self.path_pub = rospy.Publisher('traj_path', Path, queue_size=10)
        self.traj_pub = rospy.Publisher('trajectory_viz', MarkerArray, queue_size=10)
        self.obstacle_pub = rospy.Publisher('obstacles', Marker, queue_size=10)

        self.obstacles = obstacles

        self.traj_count = 0

        self.rate = rospy.Rate(1)

    def start(self):
        while not rospy.is_shutdown():
            self.marker_count = 0
            for obs in obstacles:
                self.show_obstacle(obs)
                self.marker_count += 1

            self.rate.sleep()

    def traj_cb(self, data):
        cpts = []
        for pt in data.cpts:
            cpts.append([pt.x, pt.y])
        cpts = np.array(cpts, dtype=float).T
        traj = Bernstein(cpts, data.t0, data.tf)

        self.show_trajectory(traj)

        # path = Path()
        # poses = []
        # for pt in traj(np.linspace(data.t0, data.tf, 101)).T:
        #     tmp = PoseStamped()
        #     tmp.header.frame_id = 'world'
        #     tmp.pose.position.x = pt[0]
        #     tmp.pose.position.y = pt[1]
        #     poses.append(tmp)

        # path.poses = poses
        # path.header.frame_id = 'world'
        # path.header.stamp = rospy.get_rostime()

        # self.path_pub.publish(path)

    def show_trajectory(self, trajectory):
        marker_array = MarkerArray()
        marker_list = []

        pts = trajectory(np.linspace(trajectory.t0, trajectory.tf, 101)).T
        marker_count = 0
        # for pt in pts:
        #     marker = Marker()
        #     marker.header.frame_id = 'world'
        #     marker.header.stamp = rospy.get_rostime()
        #     marker.ns = f'trajectory_{self.traj_count}'
        #     self.traj_count += 1
        #     marker.type = Marker.SPHERE
        #     marker.action = Marker.ADD
        #     marker.scale.x = 0.1
        #     marker.scale.y = 0.1 #1e-7
        #     marker.scale.z = 0.1 # 1e-7

        #     marker.id = marker_count
        #     marker_count += 1
        #     marker.pose.position.x = pt[0]
        #     marker.pose.position.y = pt[1]
        #     marker.pose.orientation.x = 0.0
        #     marker.pose.orientation.y = 0.0
        #     marker.pose.orientation.z = 0.0
        #     marker.pose.orientation.w = 1.0

        #     red, green, blue, alpha = plt.cm.jet(marker_count/len(pts))
        #     marker.color.r = red
        #     marker.color.g = green
        #     marker.color.b = blue
        #     marker.color.a = alpha # on't forget to set the alpha!

        #     marker_list.append(marker)
        for i in range(len(pts)-1):
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = rospy.get_rostime()
            marker.ns = f'trajectory_{self.traj_count}'
            self.traj_count += 1
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.1
            # marker.scale.y = 0.1  # 1e-7
            # marker.scale.z = 0.1  # 1e-7

            marker.id = marker_count
            marker_count += 1
            # marker.pose.position.x = pts[i, 0]
            # marker.pose.position.y = pts[i, 1]
            # marker.pose.position.z = 0
            # marker.pose.orientation.x = 0.0
            # marker.pose.orientation.y = 0.0
            # marker.pose.orientation.z = 0.0
            # marker.pose.orientation.w = 1.0
            marker.points = [Point(pts[i, 0], pts[i, 1], 0), Point(pts[i+1, 0], pts[i+1, 1], 0)]

            red, green, blue, alpha = plt.cm.jet(marker_count/len(pts))
            marker.color.r = red
            marker.color.g = green
            marker.color.b = blue
            marker.color.a = alpha  # on't forget to set the alpha!

            marker_list.append(marker)

        marker_array.markers = marker_list

        self.traj_pub.publish(marker_array)

        # marker.pose.position.x = position[0]
        # marker.pose.position.y = position[1]
        # marker.pose.position.z = 0
        # marker.pose.orientation.x = 0.0
        # marker.pose.orientation.y = 0.0
        # marker.pose.orientation.z = 0.0
        # marker.pose.orientation.w = 1.0
        # marker.scale.x = 2  # TODO: Define the obstacle size somewhere else so that it isn't hard coded here
        # marker.scale.y = 2
        # marker.scale.z = 2
        # marker.color.a = 1.0 # on't forget to set the alpha!
        # marker.color.r = 0.0
        # marker.color.g = 0.0
        # marker.color.b = 1.0
        # marker.lifetime = rospy.Duration(10)
        # # only if using a MESH_RESOURCE marker type:
        # # marker.mesh_resource = "package://pr2_description/meshes/base_v0/base.dae";
        # self.obstacle_pub.publish(marker)

    def show_obstacle(self, position):
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = rospy.get_rostime()
        marker.ns = 'obstacles'
        marker.id = self.marker_count
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 2  # TODO: Define the obstacle size somewhere else so that it isn't hard coded here
        marker.scale.y = 2
        marker.scale.z = 2
        marker.color.a = 1.0 # on't forget to set the alpha!
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.lifetime = rospy.Duration(10)
        # only if using a MESH_RESOURCE marker type:
        # marker.mesh_resource = "package://pr2_description/meshes/base_v0/base.dae";
        self.obstacle_pub.publish(marker)


if __name__ == '__main__':
    # TODO: Obstacles are copied and pasted here from jellyfish_controller_node.py. They should be removed from both
    # files and either added in a config file (probably yaml) or should be "discovered" using a fake perception node
    obstacles = [np.array([8, 0], dtype=float),  # Obstacle positions (m)
                 np.array([20, 2], dtype=float),
                 np.array([60, 1], dtype=float),
                 np.array([40, 2], dtype=float),
                 np.array([50, -3], dtype=float),
                 np.array([80, -3], dtype=float),
                 np.array([30, -1], dtype=float)]

    viz = Visualization()
    viz.start()
