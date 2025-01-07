import rclpy
from rclpy.node import Node
from scipy.linalg import expm
from scipy.linalg import logm
from itertools import permutations
import time
import threading
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker

import numpy as np
from youbot_kinematics.youbotKineStudent import YoubotKinematicStudent
from youbot_kinematics.target_data import TARGET_JOINT_POSITIONS

from scipy.spatial.transform import Rotation as R
from builtin_interfaces.msg import Duration

class YoubotTrajectoryPlanning(Node):
    def __init__(self):
        # Initialize node
        super().__init__('youbot_trajectory_planner')

        # Save question number for check in main run method
        self.kdl_youbot = YoubotKinematicStudent()

        # Create trajectory publisher and a checkpoint publisher to visualize checkpoints
        self.traj_pub = self.create_publisher(JointTrajectory, '/EffortJointInterface_trajectory_controller/command',
                                        5)
        self.checkpoint_pub = self.create_publisher(Marker, "checkpoint_positions", 100)

    def run(self):
        """This function is the main run function of the class. When called, it runs question 6 by calling the q6()
        function to get the trajectory. Then, the message is filled out and published to the /command topic.
        """
        print("run q6a")
        self.get_logger().info("Waiting 5 seconds for everything to load up.")
        time.sleep(2.0)
        traj = self.q6()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = ["arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5"]
        self.traj_pub.publish(traj)

    def q6(self):
        """ This is the main q6 function. Here, other methods are called to create the shortest path required for this
        question. Below, a general step-by-step is given as to how to solve the problem.
        Returns:
            traj (JointTrajectory): A list of JointTrajectory points giving the robot joint positions to achieve in a
            given time period.
        """
        # TODO: implement this
        # Steps to solving Q6.
        # 1. Load in targets from the bagfile (checkpoint data and target joint positions).
        # 2. Compute the shortest path achievable visiting each checkpoint Cartesian position.
        # 3. Determine intermediate checkpoints to achieve a linear path between each checkpoint and have a full list of
        #    checkpoints the robot must achieve. You can publish them to see if they look correct. Look at slides 39 in lecture 7
        # 4. Convert all the checkpoints into joint values using an inverse kinematics solver.
        # 5. Create a JointTrajectory message.

        # Your code starts here ------------------------------
        traj = JointTrajectory()
        target_tf, _ = self.load_targets()
        sorted_order, min_dist = self.get_shortest_path(target_tf)
        full_checkpoint_tfs = self.intermediate_tfs(sorted_order, target_tf, 4)
        self.publish_traj_tfs(full_checkpoint_tfs)
        full_checkpoint_joints = self.full_checkpoints_to_joints(full_checkpoint_tfs, self.kdl_youbot.current_joint_position)
        traj.points = []
        for i in range(full_checkpoint_joints.shape[1]):
            point = JointTrajectoryPoint()
            point.positions = full_checkpoint_joints[:, i].tolist()
            point.time_from_start = Duration(sec=1, nanosec=0)
            traj.points.append(point)

        # Your code ends here ------------------------------

        assert isinstance(traj, JointTrajectory)
        return traj

    def load_targets(self):
        """This function loads the checkpoint data from the TARGET_JOINT_POSITIONS variable. In this variable you will find each
        row has target joint positions. You need to use forward kinematics to get the goal end-effector position.
        Returns:
            target_cart_tf (4x4x5 np.ndarray): The target 4x4 homogenous transformations of the checkpoints found in the
            bag file. There are a total of 5 transforms (4 checkpoints + 1 initial starting cartesian position).
            target_joint_positions (5x5 np.ndarray): The target joint values for the 4 checkpoints + 1 initial starting
            position.
        """
        num_target_positions = len(TARGET_JOINT_POSITIONS)
        self.get_logger().info(f"{num_target_positions} target positions")
        # Initialize arrays for checkpoint transformations and joint positions
        target_joint_positions = np.zeros((5, num_target_positions + 1))
        # Create a 4x4 transformation matrix, then stack 6 of these matrices together for each checkpoint
        target_cart_tf = np.repeat(np.identity(4), num_target_positions + 1, axis=1).reshape((4, 4, num_target_positions + 1))

        # Get the current starting position of the robot
        target_joint_positions[:, 0] = self.kdl_youbot.current_joint_position
        # Initialize the first checkpoint as the current end effector position
        target_cart_tf[:, :, 0] = self.kdl_youbot.forward_kinematics(target_joint_positions[:, 0].tolist())

        # TODO: populate the transforms in the target_cart_tf object
        # populate the joint positions in the target_joint_positions object
        # Your code starts here ------------------------------
        for i in range(num_target_positions):
            target_joint_positions[:, i + 1] = TARGET_JOINT_POSITIONS[i]
            target_cart_tf[:, :, i + 1] = self.kdl_youbot.forward_kinematics(list(TARGET_JOINT_POSITIONS[i]))
        # Your code ends here ------------------------------

        self.get_logger().info(f"{target_cart_tf.shape} target poses")
        assert isinstance(target_cart_tf, np.ndarray)
        assert target_cart_tf.shape == (4, 4, num_target_positions + 1)
        assert isinstance(target_joint_positions, np.ndarray)
        assert target_joint_positions.shape == (5, num_target_positions + 1)

        return target_cart_tf, target_joint_positions
        

    def get_shortest_path(self, checkpoints_tf):
        """This function takes the checkpoint transformations and computes the order of checkpoints that results
        in the shortest overall path.
        Args:
            checkpoints_tf (np.ndarray): The target checkpoints transformations as a 4x4x5 numpy ndarray.
        Returns:
            sorted_order (np.array): An array of size 5 indicating the order of checkpoint
            min_dist:  (float): The associated distance to the sorted order giving the total estimate for travel
            distance.
        """
        num_checkpoints = checkpoints_tf.shape[2]
        # TODO: implement this method. Make it flexible to accomodate different numbers of targets.
        # Your code starts here ------------------------------
        # Sort the shortest distance and permutation here
        # Record distances to initial tf
        all_possible_paths = permutations(range(1, num_checkpoints))

        min_dist = np.inf
        sorted_order = None
        for path in all_possible_paths:
            path = [0] + list(path)
            dist = 0
            for i in range(num_checkpoints - 1):
                dist += np.linalg.norm(checkpoints_tf[:3, 3, path[i]] - checkpoints_tf[:3, 3, path[i + 1]])
            if dist < min_dist:
                min_dist = dist
                sorted_order = np.array(path)

        # Your code ends here ------------------------------
        assert isinstance(sorted_order, np.ndarray)
        assert sorted_order.shape == (num_checkpoints,)
        assert isinstance(min_dist, float)

        return sorted_order, min_dist

    def publish_traj_tfs(self, tfs):
        """This function gets a np.ndarray of transforms and publishes them in a color coded fashion to show how the
        Cartesian path of the robot end-effector.
        Args:
            tfs (np.ndarray): A array of 4x4xn homogenous transformations specifying the end-effector trajectory.
        """
        id = 0
        for i in range(0, tfs.shape[2]):
            marker = Marker()
            marker.id = id
            id += 1
            marker.header.frame_id = 'base_link'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0 + id * 0.05
            marker.color.b = 1.0 - id * 0.05
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = tfs[0, -1, i]
            marker.pose.position.y = tfs[1, -1, i]
            marker.pose.position.z = tfs[2, -1, i]
            #print(f"Publishing Position {i}: {marker.pose.position.x}, {marker.pose.position.y}, {marker.pose.position.z}")
            self.checkpoint_pub.publish(marker)

    def intermediate_tfs(self, sorted_checkpoint_idx, target_checkpoint_tfs, num_points):
        """This function takes the target checkpoint transforms and the desired order based on the shortest path sorting, 
        and calls the decoupled_rot_and_trans() function.
        Args:
            sorted_checkpoint_idx (list): List describing order of checkpoints to follow.
            target_checkpoint_tfs (np.ndarray): the state of the robot joints. In a youbot those are revolute
            num_points (int): Number of intermediate points between checkpoints.
        Returns:
            full_checkpoint_tfs: 4x4x(4xnum_points + 5) homogeneous transformations matrices describing the full desired
            poses of the end-effector position.
        """
        # TODO: implement this
        # Your code starts here ------------------------------
        num_target_positions = target_checkpoint_tfs.shape[2]
        print(f"Num target positions: {num_target_positions}")
        # for i in range(num_target_positions):
        #     print(f"Checkpoint {i}: {target_checkpoint_tfs[0, :, i]}")

        full_checkpoint_tfs = np.zeros((4, 4, 4 * num_points + 5))
        full_checkpoint_tfs[:, :, 0] = target_checkpoint_tfs[:, :, 0]

        for i in range(1, num_target_positions):
            start_index = (i - 1) * (num_points + 1)
            end_index = i * (num_points + 1)
            start_tf = full_checkpoint_tfs[:, :, start_index]
            end_tf = target_checkpoint_tfs[:, :, sorted_checkpoint_idx[i]]
            # Fill in the target checkpoints
            full_checkpoint_tfs[:, :, end_index] = end_tf
            # Fill in the intermediary checkpoints
            full_checkpoint_tfs[:, :, start_index + 1:end_index]\
                = self.decoupled_rot_and_trans(start_tf, end_tf, num_points)

        return full_checkpoint_tfs

    def decoupled_rot_and_trans(self, checkpoint_a_tf, checkpoint_b_tf, num_points):
        """This function takes two checkpoint transforms and computes the intermediate transformations
        that follow a straight line path by decoupling rotation and translation.
        Args:
            checkpoint_a_tf (np.ndarray): 4x4 transformation describing pose of checkpoint a.
            checkpoint_b_tf (np.ndarray): 4x4 transformation describing pose of checkpoint b.
            num_points (int): Number of intermediate points between checkpoint a and checkpoint b.
        Returns:
            tfs: 4x4x(num_points) homogeneous transformations matrices describing the full desired
            poses of the end-effector position from checkpoint a to checkpoint b following a linear path.
        """
        self.get_logger().info("checkpoint a")
        self.get_logger().info(str(checkpoint_a_tf))
        self.get_logger().info("checkpoint b")
        self.get_logger().info(str(checkpoint_b_tf))
        # TODO: implement this
        # Your code starts here ------------------------------
        # Decouple the rotation and translation
        tfs = np.zeros((4, 4, num_points))
        R_init = checkpoint_a_tf[:3, :3]
        p_init = checkpoint_a_tf[:3, 3]
        R_des = checkpoint_b_tf[:3, :3]
        p_des = checkpoint_b_tf[:3, 3]

        # Convert mat to quaternion
        q_init = R.from_matrix(R_init).as_quat()
        q_des = R.from_matrix(R_des).as_quat()

        # Intermediary
        for i in range(num_points):
            t = (i + 1) / (num_points + 1)
            # Use LERP to interpolate between the two quaternions
            q_inter = (1 - t) * q_init + t * q_des
            # Normalize to ensure it is a unit quaternion
            q_inter = q_inter / np.linalg.norm(q_inter)
            R_inter = R.from_quat(q_inter).as_matrix()
            p_inter = (1 - t) * p_init + t * p_des
            tfs[:, :, i] = np.block([[R_inter, p_inter.reshape(3, 1)], [0, 0, 0, 1]])
        # Your code ends here ------------------------------
        return tfs

    def full_checkpoints_to_joints(self, full_checkpoint_tfs, init_joint_position):
        """This function takes the full set of checkpoint transformations, including intermediate checkpoints, 
        and computes the associated joint positions by calling the ik_position_only() function.
        Args:
            full_checkpoint_tfs (np.ndarray, 4x4xn): 4x4xn transformations describing all the desired poses of the end-effector
            to follow the desired path.
            init_joint_position (np.ndarray):A 5x1 array for the initial joint position of the robot.
        Returns:
            q_checkpoints (np.ndarray, 5xn): For each pose, the solution of the position IK to get the joint position
            for that pose.
        """
        # TODO: Implement this
        # Your code starts here ------------------------------
        num_points = full_checkpoint_tfs.shape[2]
        q_checkpoints = np.zeros((5, num_points))
        q_checkpoints[:, 0] = init_joint_position

        for i in range(1, num_points):
            q_checkpoints[:, i], _ = self.ik_position_only(full_checkpoint_tfs[:, :, i], q_checkpoints[:, i - 1])
        # Your code ends here ------------------------------

        return q_checkpoints

    def ik_position_only(self, pose, q0, lam=0.25, num=500):
        """This function implements position only inverse kinematics.
        Args:
            pose (np.ndarray, 4x4): 4x4 transformations describing the pose of the end-effector position.
            q0 (np.ndarray, 5x1):A 5x1 array for the initial starting point of the algorithm.
        Returns:
            q (np.ndarray, 5x1): The IK solution for the given pose.
            error (float): The Cartesian error of the solution.
        """

        # TODO: Implement this
        # Some useful notes:
        # We are only interested in position control - take only the position part of the pose as well as elements of the
        # Jacobian that will affect the position of the error.

        # Your code starts here ------------------------------
        # Delta p = J(q) * delta q
        # So, delta q = J(q)^-1 * delta p
        # Keep iterating until the error is small
        p_target = pose[:3, 3]
        q = q0

        for i in range(num):
            T = self.kdl_youbot.forward_kinematics(list(q))
            p = T[:3, 3]
            J = self.kdl_youbot.get_jacobian(list(q))[:3, :]
            error = p_target - p
            q += lam * np.linalg.pinv(J) @ error

            if self.kdl_youbot.check_singularity(list(q)):
                self.get_logger().info(f"Singularity {q} reached")
                break
            if np.linalg.norm(error) < 1e-6:
                break
  
        # Your code ends here ------------------------------
        return q, error


def main(args=None):
    rclpy.init(args=args)

    youbot_planner = YoubotTrajectoryPlanning()

    youbot_planner.run()

    rclpy.spin(youbot_planner)


    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    youbot_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()