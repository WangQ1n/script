# -*- coding: utf-8 -*-
import os
import re
import json
import shutil
from ruamel.yaml import YAML

ruamel_yaml = YAML()
ruamel_yaml.preserve_quotes = True  # 保留原始格式


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
    return List_FileName


def changePoints(data):
    for shape in data["shapes"]:
        points = shape["points"]
        points_str = []
        for point in points:
            point_str = "- [" + str(point[0]) + ", " + str(point[1]) + "]" 
            points_str.append(point_str)
        shape["points"] = points_str


def readJsonsFromDir(dir):
    label_names = getFileName(dir, ".json")
    if len(label_names) != 24:
        print('warnning:', dir, 'lost', 24 - len(label_names))
    jsons = []
    for name in label_names:
        label_path = os.path.join(dir, name)
        # 读取 JSON 文件
        with open(label_path, "r") as json_file:
            raw_data = json.load(json_file)
        # print("Current name:", raw_data["imagePath"])
        jsons.append(raw_data)
    return jsons


def writeTxtFile(path, jsons):
    with open(path, "w") as file:
        for data in jsons:
            name = data["imagePath"]
            file.write(f"{name}\n")
            for shape in data["shapes"]:
                label = shape["label"]
                file.write(f"----------{label}\n")
                for point in shape["points"]:
                    file.write(f"              {'- [' + str(point[0]) + ', ' + str(point[1]) + ']' }\n")
            file.write("\n")


def loadOpencvYaml(file_path):
    # ruamel_yaml = YAML()
    """加载 OpenCV YAML 文件并去除 %YAML:1.0"""
    with open(file_path, 'r') as file:
        content = ruamel_yaml.load(file)

    return content


def save_opencv_yaml(file_path, data):
    """保存数据为 OpenCV YAML 格式，并添加 %YAML:1.0"""
    gateway = "2" + file_path.split('/')[-2][-2:]
    with open(file_path, 'w') as file:
        file.write('%YAML:1.0\n')
        file.write('---\n')
        info = 'storage_path: ' + data['storage_path'] + '\n'
        file.write(info)
        cam_classes = ['left_cam', 'right_cam']
        for cam_class in cam_classes:
            info = cam_class + ':\n    [\n'
            for cam in data[cam_class]:
                if cam == data[cam_class][-1]:
                    info = info + '        ' + cam + '\n'
                else:
                    info = info + '        ' + cam + ',\n'
            info = info + '   ]\n'
            file.write(info)
        cam_infos = list(data.items())[3:]
        cam_infos = dict(cam_infos)
        for cam_name, params in cam_infos.items():
            file.write(cam_name+': \n')
            for param_name, param_value in params.items():
                if param_name == 'regions':
                    file.write('    ' + param_name + ': \n')
                    for region in param_value:
                        file.write('        - id: ' + str(region['id']) + '\n')
                        file.write('          value: \n')
                        for pt in region['value']:
                            file.write('              - '+str(pt) + '\n')
                elif param_name == 'ip':
                    file.write('    ' + param_name + ': \'' + str(param_value.replace("205", gateway)) + '\'\n')
                elif param_name == 'id':
                    file.write('    ' + param_name + ': ' + str(param_value) + ' ' + str(params.ca.items[param_name][2].value))  # 注释自动换行，无需加 ‘\n’
                else:
                    file.write('    ' + param_name + ': ' + str(param_value) + '\n')


def writeYamlFile(path, data):
    yaml_file = loadOpencvYaml(path)
    for camera in data:
        name = camera['imagePath'].split('-')[0]
        for roi in camera['shapes']:
            label = roi['label']
            if label == 'roi0':
                roi_idx = 0
            elif label == 'roi1':
                roi_idx = 1
            else:
                continue
            points = roi['points']
            if name in yaml_file:
                yaml_file[name]['regions'][roi_idx]['value'] = points
    return yaml_file


def convert(root, in_camera_path, out_camera_path):
    jsons_data = readJsonsFromDir(root)
    txt_path = root + "/infos.txt"
    writeTxtFile(txt_path, jsons_data)
    yaml_file = writeYamlFile(in_camera_path, jsons_data)
    save_opencv_yaml(out_camera_path, yaml_file)


def batchConvert(root, camera_std_path):
    trains = []
    if len(trains) == 0:
        trains = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
    for train in trains:
        convert(os.path.join(root, train), camera_std_path, os.path.join(root, train, 'camera.yaml'))


def copy_and_rename_camera_configs(source_base_dir, target_dir="camera_config"):
    """
    将源目录中按序号命名的文件夹中的camera.yaml文件复制到目标目录，
    并按文件夹序号重命名
    
    参数:
        source_base_dir: 包含序号文件夹的源目录路径
        target_dir: 目标目录名称（默认为"camera_config"）
    """
    
    # 创建目标目录
    target_path = os.path.join(os.getcwd(), target_dir)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"创建目标目录: {target_path}")
    
    # 获取源目录下所有项目
    items = os.listdir(source_base_dir)
    
    # 用于存储序号文件夹的字典
    numbered_dirs = {}
    
    # 遍历源目录，找出以序号命名的文件夹
    for item in items:
        item_path = os.path.join(source_base_dir, item)
        
        # 只处理文件夹
        if os.path.isdir(item_path):
            # 尝试从文件夹名中提取序号
            match = re.match(r'^(\d+)', item)
            if match:
                number = match.group(1)
                numbered_dirs[number] = item_path
                print(f"找到序号文件夹: {item} (序号: {number})")
    
    # 复制并重命名文件
    copied_count = 0
    for number, dir_path in numbered_dirs.items():
        camera_yaml_path = os.path.join(dir_path, "camera.yaml")
        
        # 检查camera.yaml文件是否存在
        if os.path.exists(camera_yaml_path):
            # 构建目标文件名
            target_filename = f"camera_{number}.yaml"
            target_filepath = os.path.join(target_path, target_filename)
            
            try:
                # 复制文件
                shutil.copy2(camera_yaml_path, target_filepath)
                print(f"已复制: {camera_yaml_path} -> {target_filepath}")
                copied_count += 1
            except Exception as e:
                print(f"复制文件失败 {camera_yaml_path}: {e}")
        else:
            print(f"警告: 在文件夹 {dir_path} 中未找到 camera.yaml 文件")
    
    print(f"\n操作完成! 共成功复制 {copied_count} 个文件到 {target_path}")


def main():
    """
    二门防夹将labelme标签转为camera配置文件格式
    1.将labelme的json写入到infos.txt、更新到camera.yaml,
    注:只更新txt中存在的标签。
    """
    camera_std_path = "/home/crrcdt123/datasets2/twoDoor/二门列车标签/camera_std.yaml"  # 模板文件
    is_batch = True
    if is_batch:
        root = "/home/crrcdt123/datasets2/twoDoor/二门列车标签/"
        batchConvert(root, camera_std_path)
        copy_and_rename_camera_configs(root, os.path.join(root, 'camera_config'))
    else:
        root = "/home/crrcdt123/datasets2/twoDoor/二门列车标签/0808/"
        convert(root, camera_std_path, os.path.join(root, 'camera.yaml'))


if __name__ == '__main__':
    main()



