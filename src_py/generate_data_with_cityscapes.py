import os
import json
import glob
import cv2
from tqdm import tqdm
# 路径配置
city_img_dir = '/media/crrcdt123/glam/public_datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train'
city_ann_dir = '/media/crrcdt123/glam/public_datasets/cityscapes/gtFine_trainvaltest/gtFine/train'
save_img_dir = '/home/crrcdt123/git/script/det_dataset/cityscapes/person/images'
save_lbl_dir = '/home/crrcdt123/git/script/det_dataset/cityscapes/person/yolo'
save_json_dir = '/home/crrcdt123/git/script/det_dataset/cityscapes/person/labelme'
for d in [save_img_dir, save_lbl_dir, save_json_dir]:
    os.makedirs(d, exist_ok=True)

target_label = 'person'
target_class_id = 0

json_list = glob.glob(os.path.join(city_ann_dir, '*', '*_gtFine_polygons.json'))

for json_file in tqdm(json_list):
    with open(json_file, 'r') as f:
        data = json.load(f)

    json_filename = os.path.basename(json_file)  # aachen_000000_000019_gtFine_polygons.json
    image_file = json_filename.replace('_gtFine_polygons.json', '_leftImg8bit.png')  # aachen_000000_000019_leftImg8bit.png

    city = os.path.basename(os.path.dirname(json_file))
    img_path = os.path.join(city_img_dir, city, image_file)
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    shapes = []
    yolo_lines = []

    for obj in data['objects']:
        if obj['label'] != target_label:
            continue

        pts = obj['polygon']
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(w, max(xs)), min(h, max(ys))

        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw / 2
        cy = y1 + bh / 2
        if bw * bh < 2880:
            continue

        # YOLO 格式 bbox
        yolo_line = f"{target_class_id} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}"
        yolo_lines.append(yolo_line)

        # LabelMe 格式 bbox
        shapes.append({
            "label": "person",
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        })

    if not yolo_lines:
        continue  # 忽略无目标的图像

    # 保存图像为 .jpg
    outname = image_file.replace('.png', '.jpg')
    out_img_path = os.path.join(save_img_dir, outname)
    out_lbl_path = os.path.join(save_lbl_dir, outname.replace('.jpg', '.txt'))
    out_json_path = os.path.join(save_json_dir, outname.replace('.jpg', '.json'))

    cv2.imwrite(out_img_path, img)

    # 写 YOLO 标签
    with open(out_lbl_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

    # 写 LabelMe 标签
    json_out = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": "../images/" + outname,
        "imageHeight": h,
        "imageWidth": w,
        "imageData": None
    }
    with open(out_json_path, 'w') as jf:
        json.dump(json_out, jf, indent=4)
