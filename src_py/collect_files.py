##################################
# 整理无标签的图像：搜集匹配的图像和标签，存入另一个文件夹
##################################
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
    compare_dir = "/media/crrcdt123/glam/crrc/datasets/s8/lidar//pcd_train/"
    collect_dir = "/media/crrcdt123/glam/crrc/datasets/s8/lidar/pcd_train_intensity/"
    save_dir = "/media/crrcdt123/glam/crrc/datasets/s8/lidar/pcd_train_intensity_b/"
    file_names = getFileName(compare_dir, ".pcd")
    collect_file_names = getFileName(collect_dir, ".pcd")
    for name in file_names:
        if name in collect_file_names:
            os.system("cp " + collect_dir + name + " " + save_dir + name)
        else:
            print("file not exsit:", name)


if __name__ == '__main__':
    main()
