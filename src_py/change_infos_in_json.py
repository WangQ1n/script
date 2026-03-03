# -*- coding: utf-8 -*-
import os
import json
import numpy as np


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


def ChangePoints(data):
    scale_ration = 1000 / 100
    tf_matrix0 = np.array([[1, -6.01963905e-07, -3.07315568e-05, 3.75999999],
                          [5.91666605e-07, 0.99999994, -0.000335071818, -0.600000024],
                          [3.07317569e-05, 0.000335071789, 0.99999994, 0.000000],
                          [0, 0.000000, 0.000000, 1.000000]])
    tf_matrix = np.array([[0.999962, 0.000152299, 0.00872521, 3.76],
                          [0, 0.999848, -0.0174524, -0.6],
                          [-0.00872654, 0.0174517, 0.99981, 0],
                          [0, 0.000000, 0.000000, 1.000000]])
    inverse_transformation_matrix = np.linalg.inv(tf_matrix0)
    for shape in data["shapes"]:
        polygon = shape["points"]
        # print(type(polygon))
        polygon = np.asarray(polygon)
        polygon_cloud = np.ones((polygon.shape[0], 4))
        polygon_cloud[:, 1] = (polygon[:, 0] - 500) / scale_ration / 5
        polygon_cloud[:, 2] = (1000 - polygon[:, 1]) / scale_ration / 1
        # homogeneous_polygon = np.hstack((polygon[:, :3], np.ones((polygon.shape[0], 2))))
        restored_polygon = np.dot(polygon_cloud, inverse_transformation_matrix.T)
        homogeneous_polygon2 = np.dot(restored_polygon, tf_matrix.T)
        polygon[:, 0] = homogeneous_polygon2[:, 1] * scale_ration * 5 + 500
        polygon[:, 1] = 1000 - homogeneous_polygon2[:, 2] * scale_ration * 1
        shape["points"] = polygon.tolist()


def ChangePath(data):
    data["imagePath"] = "../images/" + os.path.basename(data["imagePath"])
    data["imageData"] = None


def main():
    """
    抽帧获取图像
    """
    root = "/media/crrcdt123/glam/crrc/data/su7/20260104/train/labels"
    label_names = getFileName(root, ".json")
    for name in label_names:
        label_path = os.path.join(root, name)
        # 读取 JSON 文件
        with open(label_path, "r") as json_file:
            data = json.load(json_file)

        # 输出当前字段值
        print("Current name:", data["imagePath"])

        # 修改字段值
        # for shape in data["shapes"]:
        #     shape["label"] = "railway"
        # ChangePoints(data)
        ChangePath(data)
        # 保存回文件
        with open(label_path, "w") as json_file:
            json.dump(data, json_file, indent=4)


if __name__ == '__main__':
    main()
