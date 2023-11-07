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
    root = "/media/crrcdt123/glam1/crrc/datasets/shen12/1240障碍物视频/1车_image/"
    convert_img_path = "/media/crrcdt123/glam1/crrc/datasets/shen12/1240障碍物视频/1车_image/convertimg"
    convert_label_path = "/media/crrcdt123/glam1/crrc/datasets/shen12/1240障碍物视频/1车_image/convertlabel"

    floder = "0021-20230620-184140"
    img_floder = os.path.join(root, floder)
    json_path = os.path.join(root, floder + "-label")
    save_img_path = os.path.join(convert_img_path, "%s_%s.jpg")
    save_label_path = os.path.join(convert_label_path, "%s_%s.json")
    label_names = getFileName(json_path, ".json")
    img_path_in_json = "../images/%s_%s.jpg"
    for name in label_names:
        label_path = os.path.join(json_path, name)
        img_path = os.path.join(img_floder, name.replace(".json", ".jpg"))
        if not os.path.exists(img_path):
            print("图像文件缺失：%s" % img_path)
            continue

        # 读取 JSON 文件
        with open(label_path, "r") as json_file:
            data = json.load(json_file)

        # 输出当前字段值
        print("Current name:", data["imagePath"])

        # 修改字段值
        data["imagePath"] = img_path_in_json % (floder, name.replace(".json", ""))

        # 保存回文件
        with open(label_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        os.system("cp " + img_path + " " + save_img_path % (floder, name.replace(".json", "")))
        os.system("cp " + label_path + " " + save_label_path % (floder, name.replace(".json", "")))

if __name__ == '__main__':
    main()
