import rclpy
from rclpy.node import Node
import numpy as np
from numpy.typing import NDArray

from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Quaternion
from trajectory_msgs.msg import JointTrajectory

# TODO: Ensure this library is findable and the method is implemented
from transform_helpers.utils import rotmat2q

class YoubotKinematicBase(Node):
    def __init__(self, tf_suffix=''):
        super().__init__('youbot_kinematic_base')
        # Robot variables
        # Identify class used when broadcasting tf with a suffix
        self.tf_suffix = tf_suffix
	
        youbot_dh_parameters = {'a': [-0.033, 0.155, 0.135, +0.002, 0.0],
                                'alpha': [np.pi / 2, 0.0, 0.0, np.pi / 2, np.pi],
                                'd': [0.145, 0.0, 0.0, 0.0, -0.185],
                                'theta': [np.pi, np.pi / 2, 0.0, -np.pi / 2, np.pi]}
        
        self.dh_params = youbot_dh_parameters.copy()

        # Set current joint position
        self.current_joint_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # Set joint limits
        self.joint_limit_min = np.array([-169 * np.pi / 180, -65 * np.pi / 180, -150 * np.pi / 180,
                                         -102.5 * np.pi / 180, -167.5 * np.pi / 180])
        self.joint_limit_max = np.array([169 * np.pi / 180, 90 * np.pi / 180, 146 * np.pi / 180,
                                         102.5 * np.pi / 180, 167.5 * np.pi / 180])
                                         

        # ROS related
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            5)
        self.joint_state_sub  # prevent unused variable warning
        self.traj_publisher = self.create_publisher(JointTrajectory, '/EffortJointInterface_trajectory_controller/command', 5)
        self.traj_publisher
        # Initialize the transform broadcaster
        self.pose_broadcaster = TransformBroadcaster(self)


    def joint_state_callback(self, msg):
        """ ROS callback function for joint states of the robot. Broadcasts the current pose of end effector.

        Args:
            msg (JointState): Joint state message containing current robot joint position.

        """
        self.current_joint_position = list(msg.position)
        current_pose = self.forward_kinematics(self.current_joint_position)
        self.broadcast_pose(current_pose)

    def broadcast_pose(self, pose):
        """Given a pose transformation matrix, broadcast the pose to the TF tree.

        Args:
            pose (np.ndarray): Transformation matrix of pose to broadcast.

        """
        transform = TransformStamped()

        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'base_link'
        transform.child_frame_id = 'arm_end_effector_' + self.tf_suffix

        transform.transform.translation.x = pose[0, 3]
        transform.transform.translation.y = pose[1, 3]
        transform.transform.translation.z = pose[2, 3]
        transform.transform.rotation = rotmat2q(pose[:3, :3])

        self.pose_broadcaster.sendTransform(transform)

    def forward_kinematics(self, joint_readings, up_to_joint=5):
        """This function solves forward kinematics by multiplying frame transformation up until a specified
        frame number. The frame transformation used in the computation are derived from dh parameters found in the
        init method and joint_readings.
        Args:
            joint_readings (list): the state of the robot joints. In a youbot those are revolute
            up_to_joint (int, optional): Specify up to what frame you want to compute forward kinematics.
                Defaults to 5.
        """

        # Transformation matrix
        T = np.eye(4)
        for i in range(up_to_joint):
            T_i = self.standard_dh(self.dh_params['a'][i], self.dh_params['alpha'][i],
                                   self.dh_params['d'][i], self.dh_params['theta'][i] + joint_readings[i])
            T = np.dot(T, T_i)

        # Sanity check
        assert isinstance(T, np.ndarray), "Output wasn't of type ndarray"
        assert T.shape == (4, 4), "Output had wrong dimensions"

        # Given the func "boradcast_pose" expects a transformation matrix, 
        # we return the transformation matrix
        return T

    def get_jacobian(self, joint):
        """Compute Jacobian given the robot joint values. Implementation found in child classes.

        Args:
            joint (list): the state of the robot joints. In a youbot those are revolute.
        Returns: Jacaobian matrix.

        """

        J = np.zeros((6, 5))
        P_e = self.forward_kinematics(joint)[:3, 3]

        for i in range(5):
            # i = 0, 1, 2, 3, 4
            T_i = self.forward_kinematics(joint, i)
            P_i_minus_1 = T_i[:3, 3]
            if i == 0:
                Z_i_minus_1 = [0, 0, -1]
            else:
                Z_i_minus_1 = T_i[:3, 2]
            

            # For revolute joints
            # J_pi-1 = z_i-1 x (p_e - p_i-1)
            # J_oi-1 = z_i-1
            J[:3, i] = np.cross(Z_i_minus_1, P_e - P_i_minus_1)
            J[3:, i] = Z_i_minus_1
            

        # Sanity check
        assert isinstance(J, np.ndarray), "Output wasn't of type ndarray"
        assert J.shape == (6, 5), "Output had wrong dimensions"  

        return J

    @staticmethod
    def standard_dh(a, alpha, d, theta):
        """This function computes the homogeneous 4x4 transformation matrix T_i based on the four standard DH parameters
         associated with link i and joint i.
        Args:
            a ([int, float]): Link Length. The distance along x_i ( the common normal) between z_{i-1} and z_i
            alpha ([int, float]): Link twist. The angle between z_{i-1} and z_i around x_i.
            d ([int, float]): Link Offset. The distance along z_{i-1} between x_{i-1} and x_i.
            theta ([int, float]): Joint angle. The angle between x_{i-1} and x_i around z_{i-1}
        Returns:
            [np.ndarray]: the 4x4 transformation matrix T_i describing  a coordinate transformation from
            the concurrent coordinate system i to the previous coordinate system i-1
        """
        assert isinstance(a, (int, float)), "wrong input type for a"
        assert isinstance(alpha, (int, float)), "wrong input type for =alpha"
        assert isinstance(d, (int, float)), "wrong input type for d"
        assert isinstance(theta, (int, float)), "wrong input type for theta"
        A = np.zeros((4, 4))

        # TODO: implement a method to get the transform matrix using DH Parameters
        # Transform matrix is of the form:
        # [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)]
        # [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)]
        # [0, sin(alpha), cos(alpha), d]
        # [0, 0, 0, 1]
        
        A[0, 0] = np.cos(theta)
        A[0, 1] = -np.sin(theta) * np.cos(alpha)
        A[0, 2] = np.sin(theta) * np.sin(alpha)
        A[0, 3] = a * np.cos(theta)

        A[1, 0] = np.sin(theta)
        A[1, 1] = np.cos(theta) * np.cos(alpha)
        A[1, 2] = -np.cos(theta) * np.sin(alpha)
        A[1, 3] = a * np.sin(theta)

        A[2, 1] = np.sin(alpha)
        A[2, 2] = np.cos(alpha)
        A[2, 3] = d
        
        A[3, 3] = 1

        assert isinstance(A, np.ndarray), "Output wasn't of type ndarray"
        assert A.shape == (4, 4), "Output had wrong dimensions"
        return A

    def rotmat2rodrigues(self, T):
        """Convert transformation matrix to rodrigues vector. Done by first converting the rotation to quaternion then
	to rodrigues.

        Args:
            T (np.ndarray): 4x4 transformation matrix to convert to pose vector - translation plus rodrigues rotation
	    representation.

        Returns:
            p (np.ndarray): An array where the first 3 elements specify the translation and the last three specify the
	    rotation.
        """
        assert isinstance(T, np.ndarray)

        # TODO: Implement a method to convert from a 4x4 transformation matrix to a translation and rodrigues vector

        p = np.empty(6, float)
        
        # Convert rotation matrix to quaternion
        q = rotmat2q(T)

        # Convert quaternion to rodrigues vector
        p[:3] = T[:3, 3]
        p[3:] = q[1:]

        return p
