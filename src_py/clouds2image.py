import numpy as np
import cv2
import matplotlib.pyplot as plt

def project_points(points, K, R, t):
    """
    Project 3D points to 2D using camera matrix, rotation matrix and translation vector.

    :param points: ndarray of shape (n, 3), 3D points
    :param K: ndarray of shape (3, 3), camera intrinsic matrix
    :param R: ndarray of shape (3, 3), rotation matrix
    :param t: ndarray of shape (3, 1), translation vector
    :return: ndarray of shape (n, 2), 2D projected points
    """
    # Convert 3D points to homogeneous coordinates
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Create the projection matrix
    P = K @ np.hstack((R, t))
    
    # Project points
    projected_points = points_homogeneous @ P.T
    
    # Convert from homogeneous to 2D coordinates
    projected_points /= projected_points[:, 2][:, np.newaxis]
    
    return projected_points[:, :2]

# Define camera intrinsic parameters
K = np.array([
    [800, 0, 320],   # fx, 0, cx
    [0, 800, 240],   # 0, fy, cy
    [0, 0, 1]        # 0, 0, 1
])

# Define rotation matrix (identity matrix, no rotation)
R = np.eye(3)

# Define translation vector (no translation)
t = np.zeros((3, 1))

# Define some 3D points
points_3D = np.array([
    [1, 1, 5],
    [2, 2, 5],
    [3, 3, 5],
    [4, 4, 5],
    [5, 5, 5]
])

# Project the points
points_2D = project_points(points_3D, K, R, t)

# Display the points
plt.scatter(points_2D[:, 0], points_2D[:, 1])
plt.xlim(0, 640)
plt.ylim(480, 0)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Projected 2D Points')
plt.show()
