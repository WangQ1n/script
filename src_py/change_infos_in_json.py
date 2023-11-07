# -*- coding: utf-8 -*-
import os
import json

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

def main():
    """
    抽帧获取图像
    """
    root = "/home/crrcdt123/datasets/segmentation/2023-08-11/labels/"
    label_names = getFileName(root, ".json")
    img_path = "../images/%s"
    for name in label_names:
        label_path = os.path.join(root, name)
        # 读取 JSON 文件
        with open(label_path, "r") as json_file:
            data = json.load(json_file)

        # 输出当前字段值
        print("Current name:", data["imagePath"])

        # 修改字段值
        for shape in data["shapes"]:
            shape["label"] = "railway"
            
        data["imagePath"] = img_path % (os.path.basename(data["imagePath"]))
        data["imageData"] = None
        # 保存回文件
        with open(label_path, "w") as json_file:
            json.dump(data, json_file, indent=4)


if __name__ == '__main__':
    main()
