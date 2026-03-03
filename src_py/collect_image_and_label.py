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
    root = "/media/crrcdt123/glam/crrc/data/su7/20260206/video_obj/"
    collect_dir = "/home/crrcdt123/datasets/railway_segmentation/su7/2026-02-06/"
    img_dir = os.path.join(root, "images")
    json_dir = os.path.join(root, "labels")
    save_img_dir = os.path.join(collect_dir, "images")
    save_img_dir2 = os.path.join(collect_dir, "train2")
    save_label_dir = os.path.join(collect_dir, "labels")
    save_img_path = os.path.join(save_img_dir, "%s")
    save_label_path = os.path.join(save_label_dir, "%s")
    for yolo_path in (save_img_dir, save_img_dir2, save_label_dir):
        if os.path.exists(yolo_path) is False:
            os.makedirs(yolo_path)
        else:
            print("dir is exist:%s" % yolo_path)

    label_names = getFileName(json_dir, ".json")
    for name in label_names:
        label_path = os.path.join(json_dir, name)
        img_path = os.path.join(img_dir, name.replace(".json", ".jpg"))
        if not os.path.exists(img_path):
            print("图像文件缺失：%s" % img_path)
            continue
        # if " (1)" in name:
        #     # 读取 JSON 文件
        #     with open(label_path, 'r') as file:
        #         data = json.load(file)
        #     # 修改 imagePath 字段信息
        #     data['imagePath'] = data['imagePath'].replace(" (1)", "")
        #     # 将修改后的数据写回 JSON 文件
        #     with open(label_path, 'w') as file:
        #         json.dump(data, file, indent=2)
        os.system("cp '" + img_path + "' " + save_img_path %
                  (name.replace(".json", ".jpg")))
        # os.system("cp '" + img_path.replace("test", "test2") + "' " + save_img_path.replace("train", "train2") %
        #           (name.replace(".json", ".jpg")))
        os.system("cp '" + label_path + "' " + save_label_path %
                  (name))


if __name__ == '__main__':
    main()
