# -*- coding: utf-8 -*-
import os
import cv2
import json
import numpy as np
import glob
import random

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


def json2mask(img_dir, json_dir, mask_dir):
    """
    将多边形生成mask图像
    """
    save_dir = mask_dir
    save_path = os.path.join(save_dir, "%s.png")
    label_names = getFileName(json_dir, ".json")

    for name in label_names:
        label_path = os.path.join(json_dir, name)
        img_path = os.path.join(img_dir, name.replace(".json", ".jpg"))
        if not os.path.exists(img_path):
            print("图像文件缺失：%s" % img_path)
            continue
        polys = readjson(label_path)
        img = cv2.imread(img_path)
        mask = np.zeros(img.shape, dtype=np.uint8)
        for poly in polys:
            poly = np.array(poly).astype(np.int)
            cv2.fillPoly(mask, [poly], (1, 1, 1), 1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cv2.imshow("img", mask)
        cv2.waitKey(30)
        cv2.imwrite(save_path % name.replace(".json", ""), mask)


def make_txt(txt_dir, image_dir, mask_dir):
    imgname_list = []
    proportion = 0.1  # val 占比
    txt_path = os.path.join(txt_dir, "%s")
    img_names = os.listdir(image_dir)
    # 复制图片
    for name in img_names:
        img_path = os.path.join(image_dir, name)
        mask_path = os.path.join(mask_dir, name.replace(".jpg", ".png"))
        if not os.path.exists(mask_path):
            print(mask_path)
            continue
        imgname_list.append(name.replace(".jpg", ""))

    # 添加txt
    file_trainval = open(txt_path % 'trainval.txt', 'w')
    for name in imgname_list:
        file_trainval.writelines(name)
        file_trainval.writelines('\n')
    file_trainval.close()

    file_train = open(txt_path % 'train.txt', 'w')
    file_val = open(txt_path % 'val.txt', 'w')
    for name in imgname_list:
        if random.random() > proportion:
            file_train.writelines(name)
            file_train.writelines('\n')
        else:
            file_val.writelines(name)
            file_val.writelines('\n')
    file_train.close()
    file_val.close()


def main():
    # root = "/home/crrcdt123/data/"
    json_dir = "/home/crrcdt123/datasets/voc/voc_label"
    txt_dir = "/home/crrcdt123/datasets/voc/ImageSets/Segmentation"
    img_dir = "/home/crrcdt123/datasets/voc/JPEGImages"
    mask_dir = "/home/crrcdt123/datasets/voc/SegmentationClass"
    json2mask(img_dir, json_dir, mask_dir)
    make_txt(txt_dir, img_dir, mask_dir)
    

if __name__ == '__main__':
    main()
