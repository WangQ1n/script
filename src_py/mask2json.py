# -*- coding: utf-8 -*-
import os
import cv2
import json
import numpy as np
import glob
import random

def decode_mask(path):
    polys = []
    if os.path.exists(path) is False:
        return polys
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img1 = img[: 540, ...]
    img2 = img[540 : 1080, ...]
    contours = []
    contours, h = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys.append(contours)
    print("轮廓数量：", len(contours))
    contours = []
    # assert(len(contours) == 1)
    contours, h = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("轮廓数量：", len(contours))
    # assert(len(contours) == 1)
    polys.append(contours)
    # print(polys)

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
        polys = readmask(label_path)
        img = cv2.imread(img_path)
        mask = np.zeros(img.shape, dtype=np.uint8)
        for poly in polys:
            poly = np.array(poly).astype(np.int)
            cv2.fillPoly(mask, [poly], (1, 1, 1), 1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cv2.imshow("img", mask)
        cv2.waitKey(30)
        cv2.imwrite(save_path % name.replace(".json", ""), mask)

def mask2json(img_dir, json_dir, mask_dir):
    """
    将多边形生成mask图像
    """
    save_dir = json_dir
    save_path = os.path.join(save_dir, "%s.json")
    mask_names = getFileName(mask_dir, ".png")

    for name in mask_names:
        mask_path = os.path.join(mask_dir, name)
        img_path = os.path.join(img_dir, name.replace(".png", ".jpg"))
        if not os.path.exists(img_path):
            print("图像文件缺失：%s" % img_path)
            continue
        print(mask_path)
        polys = decode_mask(mask_path)
        img = cv2.imread(img_path)
        # mask = np.zeros(img.shape, dtype=np.uint8)
        img1 = img[: 540, ...]
        img2 = img[540 : 1080, ...]
        for i in range(len(polys[0])):
            cv2.drawContours(img1, polys[0], i, (0, 0, 255), 2)
        for i in range(len(polys[1])):
            cv2.drawContours(img2, polys[1], i, (0, 0, 255), 2)
        mask = img
        cv2.imshow("img", mask)
        cv2.waitKey(0)
        # cv2.imwrite(save_path % name.replace(".png", ""), mask)

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
    root = "/home/crrcdt123/pytorch_test/pytorch-deeplab-xception/traindata/train_voc/"
    json_dir = root + "/json/"
    # txt_dir = "/home/crrcdt123/datasets/voc/ImageSets/Segmentation"
    img_dir = root + "/JPEGImages"
    mask_dir = root + "/SegmentationClass"
    mask2json(img_dir, json_dir, mask_dir)
    # make_txt(txt_dir, img_dir, mask_dir)
    

if __name__ == '__main__':
    main()
