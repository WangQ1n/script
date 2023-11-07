# -*- coding: utf-8 -*-
import os
import cv2
import json
import numpy as np


def readjson(path):
    # 读取xml文件
    polys = []
    if os.path.exists(path) is False:
        return objects
    
    with open(path, 'r', encoding = 'utf-8') as fp:
        data = json.load(fp)
        for shape in data["shapes"]:
            polys.append(shape["points"])
        # print(data)

    return polys


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
    将多边形生成mask图像
    """
    root = "/home/crrcdt123/data/seg_data"
    save_dir = os.path.join("/home/crrcdt123/data", "seg_mask")
    save_path = os.path.join(save_dir, "%s.png")
    label_names = getFileName(root, ".json")

    for name in label_names:
        label_path = os.path.join(root, name)
        img_path = label_path.replace(".json", ".jpg")
        if not os.path.exists(img_path):
            print("图像文件缺失：%s" % img_path)
            continue
        polys = readjson(label_path)
        img = cv2.imread(img_path)
        mask = np.zeros(img.shape, dtype=np.uint8)
        for poly in polys:
            poly = np.array(poly).astype(np.int)
            cv2.fillPoly(mask, [poly], (1, 1, 1), 1)

        cv2.imshow("img", mask)
        cv2.waitKey(30)
        cv2.imwrite(save_path % name, mask)
    # print(img, max(img[:, ]))
    

if __name__ == '__main__':
    main()
