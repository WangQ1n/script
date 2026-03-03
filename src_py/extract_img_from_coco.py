import os
import random
import shutil
from pycocotools.coco import COCO
from tqdm import tqdm
import json
# ===== 参数配置 =====
coco_json = '/media/crrcdt123/glam/public_datasets/coco/annotations/instances_train2017.json'
coco_img_dir = '/media/crrcdt123/glam/public_datasets/coco/train2017'
output_img_dir = '/media/crrcdt123/glam/public_datasets/coco/images'
output_yolo_dir = '/media/crrcdt123/glam/public_datasets/coco/yolo'
output_labelme_dir = '/media/crrcdt123/glam/public_datasets/coco/labelme'
target_class_name = 'person'
area_thresh = 1  # person bbox面积大于此阈值才保留
target_count = 7000

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_yolo_dir, exist_ok=True)
os.makedirs(output_labelme_dir, exist_ok=True)
# ===== 加载COCO =====
coco = COCO(coco_json)
cat_ids = coco.getCatIds(catNms=[target_class_name])
img_ids = coco.getImgIds(catIds=cat_ids)

# ===== 筛选符合条件的图像 =====
qualified_imgs = []

for img_id in tqdm(img_ids, desc="Filtering"):
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    good_anns = [ann for ann in anns if ann['area'] > area_thresh]

    if len(good_anns) >= 1:
        qualified_imgs.append((img_id, good_anns))

# ===== 随机选择 N 张图像 =====
random.shuffle(qualified_imgs)
selected = qualified_imgs[:target_count]

# ===== 生成图像与标签文件 =====
for img_id, anns in tqdm(selected, desc="Saving"):
    img_info = coco.loadImgs([img_id])[0]
    filename = img_info['file_name']
    width, height = img_info['width'], img_info['height']
    
    # 拷贝图像
    src_path = os.path.join(coco_img_dir, filename)
    dst_path = os.path.join(output_img_dir, filename)
    shutil.copy(src_path, dst_path)
    
    # 生成YOLO标签
    label_path = os.path.join(output_yolo_dir, os.path.splitext(filename)[0] + '.txt')
    with open(label_path, 'w') as f:
        for ann in anns:
            if ann['area'] < area_thresh:
                continue
            x, y, w, h = ann['bbox']
            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            bw = w / width
            bh = h / height
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    
    # 生成 LabelMe 标签
    json_out = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": "../images/" + filename,
        "imageHeight": height,
        "imageWidth": width,
        "imageData": None,
    }

    for ann in anns:
        if ann['area'] < area_thresh:
            continue
        x, y, w, h = ann['bbox']
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        shape = {
            "label": "person",
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        json_out["shapes"].append(shape)

    # 保存 json 文件
    json_path = os.path.join(output_labelme_dir, os.path.splitext(filename)[0] + ".json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as jf:
        json.dump(json_out, jf, indent=4)
