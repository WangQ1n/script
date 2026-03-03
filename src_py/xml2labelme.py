import os
import json
import xml.etree.ElementTree as ET
from PIL import Image

voc_xml_dir = '/home/crrcdt123/git/script/output/images/label'
image_dir = '/home/crrcdt123/git/script/output/images/image'
labelme_json_dir = '/home/crrcdt123/git/script/output/images/labelme_json'

os.makedirs(labelme_json_dir, exist_ok=True)

for xml_file in os.listdir(voc_xml_dir):
    if not xml_file.endswith('.xml'):
        continue

    tree = ET.parse(os.path.join(voc_xml_dir, xml_file))
    root = tree.getroot()

    image_filename = root.find('filename').text
    image_path = os.path.join(image_dir, image_filename)
    if not os.path.exists(image_path):
        print(f"[警告] 图像文件不存在: {image_path}")
        continue

    image = Image.open(image_path)
    width, height = image.size

    shapes = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        shape = {
            "label": label,
            "points": [[xmin, ymin], [xmax, ymax]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        shapes.append(shape)

    labelme_json = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": "../image/" + image_filename,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    json_path = os.path.join(labelme_json_dir, os.path.splitext(xml_file)[0] + '.json')
    with open(json_path, 'w') as f:
        json.dump(labelme_json, f, indent=2)

    print(f"✅ 已生成: {json_path}")
