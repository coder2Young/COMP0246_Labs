from geometry_msgs.msg import Quaternion

import numpy as np
from numpy.typing import NDArray

def rotmat2q(T: NDArray) -> Quaternion:
    # Function that transforms a 3x3 rotation matrix to a ros quaternion representation
    

    if T.shape != (3, 3):
        raise ValueError

    # Transform the rotation matrix to a quaternion
    q = Quaternion()
    tr = T[0,0] + T[1,1] + T[2,2]

    if tr > 0:
        S = np.sqrt(tr+1.0) * 2
        q.w = 0.25 * S
        q.x = (T[2,1] - T[1,2]) / S
        q.y = (T[0,2] - T[2,0]) / S
        q.z = (T[1,0] - T[0,1]) / S
    elif (T[0,0] > T[1,1]) and (T[0,0] > T[2,2]):
        S = np.sqrt(1.0 + T[0,0] - T[1,1] - T[2,2]) * 2
        q.w = (T[2,1] - T[1,2]) / S
        q.x = 0.25 * S
        q.y = (T[0,1] + T[1,0]) / S
        q.z = (T[0,2] + T[2,0]) / S
    elif T[1,1] > T[2,2]:
        S = np.sqrt(1.0 + T[1,1] - T[0,0] - T[2,2]) * 2
        q.w = (T[0,2] - T[2,0]) / S
        q.x = (T[0,1] + T[1,0]) / S
        q.y = 0.25 * S
        q.z = (T[1,2] + T[2,1]) / S
    else:
        S = np.sqrt(1.0 + T[2,2] - T[0,0] - T[1,1]) * 2
        q.w = (T[1,0] - T[0,1]) / S
        q.x = (T[0,2] + T[2,0]) / S
        q.y = (T[1,2] + T[2,1]) / S
        q.z = 0.25 * S

    return q