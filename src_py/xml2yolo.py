import os
import xml.etree.ElementTree as ET

# 类别映射
CLASS_NAME_TO_ID = {'person': 0, 'train': 6}

# 输入输出路径
xml_folder = '/home/crrcdt123/git/script/det_dataset/shenzhen/xml'      # VOC格式xml文件夹
output_folder = '/home/crrcdt123/git/script/det_dataset/shenzhen/yolo'   # YOLO格式txt输出文件夹
os.makedirs(output_folder, exist_ok=True)

for xml_file in os.listdir(xml_folder):
    if not xml_file.endswith('.xml'):
        continue

    tree = ET.parse(os.path.join(xml_folder, xml_file))
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    txt_lines = []

    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in CLASS_NAME_TO_ID:
            continue  # 跳过未知类别

        cls_id = CLASS_NAME_TO_ID[cls_name]
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # 转为YOLO格式
        x_center = (xmin + xmax) / 2.0 / w
        y_center = (ymin + ymax) / 2.0 / h
        box_width = (xmax - xmin) / w
        box_height = (ymax - ymin) / h

        txt_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    # 写入 .txt 文件
    base_name = os.path.splitext(xml_file)[0]
    with open(os.path.join(output_folder, base_name + '.txt'), 'w') as f:
        f.write('\n'.join(txt_lines))
