import os
import cv2
import json
# 参数
input_image_dir = '/home/crrcdt123/git/script/det_dataset/su8_dark//images'
input_label_dir = '/home/crrcdt123/git/script/det_dataset/su8_dark//yolo'
output_image_dir = '/home/crrcdt123/git/script/det_dataset/su8_roi_dark/images'
output_label_dir = '/home/crrcdt123/git/script/det_dataset/su8_roi_dark/yolo'
output_labelme_dir = '/home/crrcdt123/git/script/det_dataset/su8_roi_dark/labels_labelme'
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)
os.makedirs(output_labelme_dir, exist_ok=True)
# 裁剪区域 (x, y, w, h)600, 50, 640, 640
crop_x, crop_y, crop_w, crop_h = 600, 50, 640, 640

def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return x1, y1, x2, y2

def remap_and_convert():
    for fname in os.listdir(input_image_dir):
        if not fname.endswith(('.jpg', '.png')):
            continue
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(input_image_dir, fname)
        label_path = os.path.join(input_label_dir, base + '.txt')

        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        crop = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        if crop.shape[0] < crop_h or crop.shape[1] < crop_w:
            continue

        new_yolo_lines = []
        labelme_shapes = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])

                    x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, W, H)

                    # 是否与裁剪区域有交集
                    if x2 < crop_x or x1 > crop_x + crop_w or y2 < crop_y or y1 > crop_y + crop_h:
                        continue

                    # 裁剪区域内坐标
                    new_x1 = max(x1, crop_x) - crop_x
                    new_y1 = max(y1, crop_y) - crop_y
                    new_x2 = min(x2, crop_x + crop_w) - crop_x
                    new_y2 = min(y2, crop_y + crop_h) - crop_y

                    # YOLO 格式归一化
                    new_xc = ((new_x1 + new_x2) / 2) / crop_w
                    new_yc = ((new_y1 + new_y2) / 2) / crop_h
                    new_bw = (new_x2 - new_x1) / crop_w
                    new_bh = (new_y2 - new_y1) / crop_h

                    if new_bw <= 0 or new_bh <= 0:
                        continue

                    new_yolo_lines.append(f"{cls_id} {new_xc:.6f} {new_yc:.6f} {new_bw:.6f} {new_bh:.6f}")

                    # LabelMe 格式
                    labelme_shapes.append({
                        "label": str(cls_id),
                        "points": [[new_x1, new_y1], [new_x2, new_y2]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    })

        # 保存裁剪图像
        out_img_path = os.path.join(output_image_dir, 'roi_' + fname)
        cv2.imwrite(out_img_path, crop)

        # 保存 YOLO 标签
        out_txt_path = os.path.join(output_label_dir, 'roi_' + base + '.txt')
        with open(out_txt_path, 'w') as f:
            f.write('\n'.join(new_yolo_lines))

        # 保存 LabelMe JSON
        out_json_path = os.path.join(output_labelme_dir, 'roi_' + base + '.json')
        labelme_json = {
            "version": "5.0.1",
            "flags": {},
            "shapes": labelme_shapes,
            "imagePath": "../images/roi_" + fname,
            "imageData": None,
            "imageHeight": crop.shape[0],
            "imageWidth": crop.shape[1]
        }
        with open(out_json_path, 'w') as f:
            json.dump(labelme_json, f, indent=2)

        print(f"✅ 生成完成：{fname}")

remap_and_convert()