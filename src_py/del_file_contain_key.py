# -*- coding: utf-8 -*-
import os
import cv2


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
    root = "/home/crrcdt123/datasets2/二门防夹数据/s8-20240307_src/"
    file_names = getFileName(root, ".jpg")
    valid_names = []
    for name in file_names:
        if 'Result' in name:
            os.system("rm " + root + "/" + name)
        else:
            valid_names.append(name)

    for name in valid_names:
        path = os.path.join(root, name)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, [960, 640])
        cv2.imshow("name", img)
        key = cv2.waitKey(0)
        # cv2.destroyWindow(name) 
        if key == ord('d'):  # 如果按下的是'd'键
            os.system("rm " + root + "/" + name)

if __name__ == '__main__':
    main()
