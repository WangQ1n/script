import cv2
import numpy as np
# 定义原始图像中的四个点
pts_dst = np.array([[0.23, 39.3], [0.03, 30.5], [0.715, 20.1], [-1.22, 26.7]])
# 定义目标图像中的四个点
pts_src = np.array([[941, 482], [892, 569], [973, 761], [742, 631]])
# 计算透视变换矩阵
H, status = cv2.findHomography(pts_src, pts_dst)
print("Perspective Transform Matrix: \n", H)
# 0.63 27, 0.5, 58
points = np.array([[961, 615], [935, 407]], dtype='float32')

# 将点转换为齐次坐标
points_homogeneous = np.array([points[:, 0], points[:, 1], np.ones(points.shape[0])])

# 应用单应性矩阵进行变换
transformed_points_homogeneous = np.dot(H, points_homogeneous)

# 转换回非齐次坐标
transformed_points = transformed_points_homogeneous[:2] / transformed_points_homogeneous[2]

# 打印变换后的点
print("Transformed Points:\n", transformed_points.T)