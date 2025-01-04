import rclpy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.node import Node
import numpy as np
from tf2_ros import TransformBroadcaster
from youbot_kinematics.youbotKineStudent import YoubotKinematicStudent
from geometry_msgs.msg import TransformStamped
from transform_helpers.utils import rotmat2q
import time

# This is a ROS node that listen to trajectory and publish transforme matrix of end effector
class TrajectoryFollower(Node):
    def __init__(self):
        super().__init__('trajectory_follower')
        self.subscription = self.create_subscription(
            JointTrajectory,
            '/EffortJointInterface_trajectory_controller/command',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.tf_broadcaster = TransformBroadcaster(self)
        self.prefix = ""
        self.kdl_youbot = YoubotKinematicStudent()

    def listener_callback(self, msg: JointTrajectory):
        #self.get_logger().info("Received trajectory message: {}".format(msg))
        
        for point in msg.points:
            joint_angles = point.positions
            time_from_start = point.time_from_start

            tf = self.kdl_youbot.forward_kinematics(list(joint_angles))
            tf_msg = self.build_tf_msg(tf, time_from_start)
            self.tf_broadcaster.sendTransform(tf_msg)
            # Wait for the time from start
            time.sleep(time_from_start.sec + time_from_start.nanosec / 1e9)

    def build_tf_msg(self, transform_matrix, time_from_start):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = self.prefix + "base_link"
        tf_msg.child_frame_id = self.prefix + "end_effector"
        tf_msg.transform.translation.x = transform_matrix[0, 3]
        tf_msg.transform.translation.y = transform_matrix[1, 3]
        tf_msg.transform.translation.z = transform_matrix[2, 3]
        quat = rotmat2q(transform_matrix[:3, :3])
        tf_msg.transform.rotation.x = quat.x
        tf_msg.transform.rotation.y = quat.y
        tf_msg.transform.rotation.z = quat.z
        tf_msg.transform.rotation.w = quat.w

        return tf_msg
        
def main(args=None):
    rclpy.init(args=args)
    trajectory_follower = TrajectoryFollower()
    rclpy.spin(trajectory_follower)
    rclpy.shutdown()