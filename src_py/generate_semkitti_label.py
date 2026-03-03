import os
import json
import cv2
import numpy as np
import open3d as o3d

def getFileName(Path_File, Type_File):
    # Path_File是文件夹的路径，Type_File是文件的种类（Eg：‘.txt’）
    List_FileName = []
    # List_FileName用来存储符合筛选条件的文件的名称
    List_AllFile = os.listdir(Path_File)
    # 返回文件夹中所有文件的列表
    for i in range(len(List_AllFile)):
        # 逐个文件循环
        if os.path.splitext(List_AllFile[i])[1] == Type_File:
            # 如果第i个文件的类别是Type_File，则进入if
            List_FileName.append(List_AllFile[i])
            # 存储第i个文件的名称
            # 欢迎使用，使用请注明来源，作者姜正，邮箱jiangzheng221@mails.ucas.ac.cn，中国科学院海洋所
    return List_FileName


def read_ori_pcd(input_path):
    lidar = []
    with open(input_path, 'r') as f:
        line = f.readline().strip()
        while line:
            linestr = line.split(" ")
            if len(linestr) == 4:
                linestr_convert = list(map(float, linestr))
                lidar.append(linestr_convert)
            line = f.readline().strip()
    return np.array(lidar)


def read_labeled_pcd(filepath):
    lidar = []
    with open(filepath, 'r') as f:
        line = f.readline().strip()
        while line:
            linestr = line.split(" ")
            if len(linestr) == 5:
                linestr_convert = list(map(float, linestr))
                lidar.append(linestr_convert)
            line = f.readline().strip()
    return np.array(lidar)


def convert2bin(input_pcd_dir, output_bin_dir):
    file_list = os.listdir(input_pcd_dir)
    if not os.path.exists(output_bin_dir):
        os.makedirs(output_bin_dir)
    for file in file_list:
        (filename, extension) = os.path.splitext(file)
        velodyne_file = os.path.join(input_pcd_dir, filename) + '.pcd'
        p_xyzi = read_ori_pcd(velodyne_file)
        p_xyzi = p_xyzi.reshape((-1, 4)).astype(np.float32)
        min_val = np.amin(p_xyzi[:, 3])
        max_val = np.amax(p_xyzi[:, 3])
        p_xyzi[:, 3] = (p_xyzi[:, 3] - min_val)/(max_val-min_val)
        p_xyzi[:, 3] = np.round(p_xyzi[:, 3], decimals=2)
        p_xyzi[:, 3] = np.minimum(p_xyzi[:, 3], 0.99)
        velodyne_file_new = os.path.join(output_bin_dir, filename) + '.bin'
        p_xyzi.tofile(velodyne_file_new)


def convert2label(input_pcd_dir, output_label_dir):
    file_list = os.listdir(input_pcd_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)
    for file in file_list:
        (filename, extension) = os.path.splitext(file)
        velodyne_file = os.path.join(input_pcd_dir, filename) + '.label'
        p_xyz_label_object = read_labeled_pcd(velodyne_file)
        p_xyz_label_object = p_xyz_label_object.reshape((-1, 5))
        label = p_xyz_label_object[:, 3].astype(np.int32)
        label = label.reshape(-1)
        velodyne_file_new = os.path.join(output_label_dir, filename) + '.label'
        label.tofile(velodyne_file_new)


def convert(ori_pcd_dir, ori_bin_dir, labeled_pcd_dir, labeled_bin_dir):
    convert2bin(ori_pcd_dir, ori_bin_dir)    
    # convert2label(labeled_pcd_dir, labeled_bin_dir)


def generate(json_dir, cloud_dir, semlabel_dir, transform_cloud_dir):
    label_names = getFileName(json_dir, ".json")
    # transform_matrix = np.eye(4)
    # tf_matrix = np.array([[0.999998, -0.000034, -0.001761, 3.760000],
    #                       [0.000000, 0.999816, -0.019197, -0.600000],
    #                       [0.001761, 0.019197, 0.999814, 0.000000],
    #                       [0.000000, 0.000000, 0.000000, 1.000000]])
    # 0.999962 0.000152299  0.00872521        3.76
    #     0    0.999848  -0.0174524        -0.6
    # -0.00872654   0.0174517     0.99981           0
    #     0           0           0           1
    # tf_matrix0 = np.array([[1, -6.01963905e-07, -3.07315568e-05, 3.75999999],
    #                       [5.91666605e-07, 0.99999994, -0.000335071818, -0.600000024],
    #                       [3.07317569e-05, 0.000335071789, 0.99999994, 0.000000],
    #                       [0, 0.000000, 0.000000, 1.000000]])
    tf_matrix = np.array([[0.999962, 0.000152299, 0.00872521, 3.76],
                          [0, 0.999848, -0.0174524, -0.6],
                          [-0.00872654, 0.0174517, 0.99981, 0],
                          [0, 0.000000, 0.000000, 1.000000]])
    # inverse_transformation_matrix = np.linalg.inv(tf_matrix0)
    scale_ration = 1000 / 100
    for name in label_names:
        cloud_name = os.path.join(cloud_dir, name.replace(".json", ".pcd"))
        if not os.path.exists(cloud_name):
            continue
        label_name = os.path.join(semlabel_dir, name.replace(".json", ".label"))
        transform_cloud_path = os.path.join(transform_cloud_dir, name.replace(".json", ".pcd"))
        src_cloud = read_ori_pcd(cloud_name)
        # 将点云矩阵扩展为齐次坐标
        homogeneous_point_cloud = np.hstack((src_cloud[:, :3], np.ones((src_cloud.shape[0], 1))))
        # 对点云进行变换
        transformed_point_cloud = np.dot(homogeneous_point_cloud, tf_matrix.T)
        # 去除变换后点云的齐次坐标
        tf_cloud = src_cloud.copy()
        tf_cloud[:, :3] = transformed_point_cloud[:, :3]
        label_path = os.path.join(json_dir, name)

        # 读取 JSON 文件
        polygon = []
        with open(label_path, "r") as json_file:
            data = json.load(json_file)
            if len(data["shapes"]) != 1:
                continue
            polygon = data["shapes"][0]["points"]
        polygon = np.asarray(polygon, dtype=np.int32)
        # polygon_cloud = np.ones((polygon.shape[0], 4))
        # polygon_cloud[:, 1] = (polygon[:, 0] - 500) / scale_ration / 5
        # polygon_cloud[:, 2] = (1000 - polygon[:, 1]) / scale_ration / 1
        # # homogeneous_polygon = np.hstack((polygon[:, :3], np.ones((polygon.shape[0], 2))))
        # restored_polygon = np.dot(polygon_cloud, inverse_transformation_matrix.T)
        # homogeneous_polygon2 = np.dot(restored_polygon, tf_matrix.T)
        # polygon[:, 0] = homogeneous_polygon2[:, 1] * scale_ration * 5 + 500
        # polygon[:, 1] = 1000 - homogeneous_polygon2[:, 2] * scale_ration * 1
        selected_indices = []
        selected_points = []
        for i in range(tf_cloud.shape[0]):
            point = tf_cloud[i]
            if -2.5 <= point[0] <= 2.5:
                x = point[1] * scale_ration * 5
                y = point[2] * scale_ration * 1
                if x > 1000 or y > 1000:
                    continue
                x = x + 1000 / 2
                y = 1000 - y
                result = cv2.pointPolygonTest(contour=polygon, pt=(x, y), measureDist=False)
                if result >= 0:
                    selected_indices.append(i)
                    selected_points.append([x, y])
        img = np.zeros((1000, 1000), dtype=np.uint8)
        cv2.polylines(img, [polygon], True, 255)
        for pt in selected_points:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 2, 255, -1)
        cv2.imshow("img", img)
        cv2.waitKey(30)
        # 创建新的点云，只包含在正负2.5米范围内的点
        # clouds = tf_cloud.extract(selected_indices)

        # 输出当前字段值
        print("Current name:", data["imagePath"])
        # 保存回文件
        if False:
            with open(label_name, "w") as f:
                f.write("VERSION .7\n")
                f.write("FIELDS x y z label object\n")
                f.write("SIZE 4, 4, 4, 4, 4\n")
                f.write("TYPE F F F I I\n")
                f.write("COUNT 1 1 1 1 1\n")
                f.write("WIDTH " + str(tf_cloud.shape[0]) + "\n")
                f.write("HEIGHT 1\n")
                f.write("POINTS " + str(tf_cloud.shape[0]) + "\n")
                f.write("VIEWPOINTS 0 0 0 1 0 0 0\n")
                f.write("DATA ascii\n")
                for i in range(tf_cloud.shape[0]):
                    point = tf_cloud[i]
                    x, y, z, intensity = point
                    if i in selected_indices:
                        # 将点云数据以二进制形式写入文件
                        f.write(str(x) + " " + str(y) + " " + str(z) + " 1 -1\n")
                    else:
                        f.write(str(x) + " " + str(y) + " " + str(z) + " 0 -1\n")
            with open(transform_cloud_path, "w") as f:
                f.write("# .PCD v0.7 - Point Cloud Data file format\n")
                f.write("VERSION 0.7\n")
                f.write("FIELDS x y z intensity\n")
                f.write("SIZE 4 4 4 4\n")
                f.write("TYPE F F F F\n")
                f.write("COUNT 1 1 1 1\n")
                f.write("WIDTH " + str(tf_cloud.shape[0]) + "\n")
                f.write("HEIGHT 1\n")
                f.write("VIEWPOINTS 0 0 0 1 0 0 0\n")
                f.write("POINTS " + str(tf_cloud.shape[0]) + "\n")
                f.write("DATA ascii\n")
                for i in range(tf_cloud.shape[0]):
                    point = tf_cloud[i]
                    x, y, z, intensity = point
                    # if i in selected_indices:
                    #     # 将点云数据以二进制形式写入文件
                    #     f.write(str(x) + " " + str(y) + " " + str(z) + " 255\n")
                    # else:
                    #     f.write(str(x) + " " + str(y) + " " + str(z) + " 1\n")
                    f.write(str(x) + " " + str(y) + " " + str(z) + " " + str(int(intensity)) + "\n")

def main():
    json_dir = "/media/crrcdt123/glam/crrc/datasets/s8/lidar/pcd_train_intensity_b/labels/"
    cloud_dir = "/media/crrcdt123/glam/crrc/datasets/s8/lidar/pcd_train_intensity_b/clouds/"
    semlabel_dir = "/media/crrcdt123/glam/crrc/datasets/s8/lidar/pcd_train_intensity_b/semlabel/"
    transform_cloud_dir = "/media/crrcdt123/glam/crrc/datasets/s8/lidar/pcd_train_intensity_b/pcd_transform/"
    ori_bin_dir = "/media/crrcdt123/glam/crrc/datasets/s8/lidar/pcd_train_intensity_b/pcd_transform_bin/"
    labeled_bin_dir = "/media/crrcdt123/glam/crrc/datasets/s8/lidar/pcd_train_intensity_b/semlabel_bin/"
    # file_names = os.listdir(labeled_bin_dir)
    # for file_name in file_names:
    #     print(file_name)
    generate(json_dir, cloud_dir, semlabel_dir, transform_cloud_dir)
    # convert(transform_cloud_dir, ori_bin_dir, semlabel_dir, labeled_bin_dir)


if __name__ == '__main__':
    main()