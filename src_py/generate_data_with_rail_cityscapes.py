import os
import cv2
import numpy as np
import random
from ultralytics import YOLO
import json
# 路径配置
track_model_path = '/home/crrcdt123/git/ultralytics_v8/runs/segment/rail/train_20240329/weights/best.pt'
track_img_dir = '/media/crrcdt123/glam/crrc/data/qiaolin/tmp/'
city_img_root = '/media/crrcdt123/glam/public_datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train'
city_ins_root = '/media/crrcdt123/glam/public_datasets/cityscapes/gtFine_trainvaltest/gtFine/train'
save_img_dir = '/home/crrcdt123/git/script/det_dataset/cityscapes/rail/images'
save_label_dir = '/home/crrcdt123/git/script/det_dataset/cityscapes/rail/yolo'
labelme_dir = '/home/crrcdt123/git/script/det_dataset/cityscapes/rail/labelme'
os.makedirs(save_img_dir, exist_ok=True)
os.makedirs(save_label_dir, exist_ok=True)
os.makedirs(labelme_dir, exist_ok=True)
# 加载轨道分割模型
track_model = YOLO(track_model_path)

def match_brightness(obj_img, bg_patch):
    mean_obj = np.mean(obj_img[obj_img > 0])
    mean_bg = np.mean(bg_patch[bg_patch > 0])
    ratio = mean_bg / (mean_obj + 1e-5)
    return np.clip(obj_img * ratio, 0, 255).astype(np.uint8)

def extract_person_instances(img_path, inst_path):
    img = cv2.imread(img_path)
    inst_mask = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED)
    person_instances = []
    for inst_id in np.unique(inst_mask):
        if inst_id // 1000 == 24:  # person 类别
            mask = (inst_mask == inst_id).astype(np.uint8)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels > 2:
                continue  # ❌ 超过1个连通区域，剔除
            if cv2.countNonZero(mask) < 2000:
                continue
            x, y, w, h = cv2.boundingRect(mask)
            crop = img[y:y+h, x:x+w]
            crop_mask = mask[y:y+h, x:x+w]
            alpha = (crop_mask * 255).astype(np.uint8)
            rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = alpha
            person_instances.append((rgba, w, h))
    return person_instances

def get_track_mask(image):
    results = track_model(image, task="segment")[0]
    for i, cls_id in enumerate(results.boxes.cls):
        if int(cls_id) == 0:  # 轨道类别为0
            return results.masks.data[i].cpu().numpy().astype(np.uint8)
    return None

def sample_points_near_track_lr(mask, sample_num=3, range_px=40):
    h, w = mask.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (range_px, 1))  # 水平膨胀
    dilated = cv2.dilate(mask, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, range_px))  # 水平膨胀
    dilated = cv2.dilate(dilated, kernel, iterations=1)
    expanded_mask = np.clip(mask + dilated, 0, 1)
    edge_mask = (expanded_mask == 1) & (mask == 0)
    yx = np.argwhere(edge_mask > 0)
    if len(yx) == 0:
        return []
    chosen = yx[np.random.choice(len(yx), min(sample_num, len(yx)), replace=False)]
    return [(int(x), int(y)) for y, x in chosen]


def paste_person(bg, person_rgba, x, y, scale=1.0):
    ph, pw = person_rgba.shape[:2]
    person_resized = cv2.resize(person_rgba, (int(pw*scale), int(ph*scale)))
    h, w = person_resized.shape[:2]

    # 🚩 将点 (x, y) 作为左下角，而不是左上角
    x = int(np.clip(x, 0, bg.shape[1] - w))
    y = int(np.clip(y - h, 0, bg.shape[0] - h))  # ⚠️ y 减去贴图高度

    overlay = person_resized[:, :, :3]
    mask = person_resized[:, :, 3]
    mask_inv = cv2.bitwise_not(mask)
    roi = bg[y:y+h, x:x+w]
    bg_roi = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg_roi = cv2.bitwise_and(overlay, overlay, mask=mask)
    cv2.copyTo(fg_roi, mask, roi)

    # 返回 YOLO 格式归一化 bbox
    cx = (x + w/2) / bg.shape[1]
    cy = (y + h/2) / bg.shape[0]
    cw = w / bg.shape[1]
    ch = h / bg.shape[0]
    return (0, cx, cy, cw, ch)  # 类别 0：person

def visualize_result(img, bboxes, paste_points):
    vis = img.copy()
    # 画每个人的bbox（归一化坐标转像素）
    for cls, cx, cy, w, h in bboxes:
        x1 = int((cx - w/2) * vis.shape[1])
        y1 = int((cy - h/2) * vis.shape[0])
        x2 = int((cx + w/2) * vis.shape[1])
        y2 = int((cy + h/2) * vis.shape[0])
        color = (0, 255, 0)  # 绿色框
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    # 画粘贴点
    for (x, y) in paste_points:
        cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)  # 红色圆点

    return vis

def visualize_track_mask(img, mask, color=(0, 255, 0), alpha=0.1):
    """
    img: 原图（BGR）
    mask: 轨道区域二值掩码（0/1）
    color: 叠加色
    alpha: 透明度
    """
    overlay = img.copy()
    overlay[mask == 1] = color
    output = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return output

def scale_by_y(y, h_img, min_scale=0.2, max_scale=1.2):
    relative_pos = y / h_img
    scale = min_scale + (relative_pos) * (max_scale - min_scale)
    return scale

# 获取 cityscapes 样本列表
city_dirs = [os.path.join(city_img_root, d) for d in os.listdir(city_img_root)]
city_images = []
for d in city_dirs:
    files = [f for f in os.listdir(d) if f.endswith('.png')]
    for f in files:
        city_images.append((os.path.join(d, f),
                            os.path.join(d.replace('leftImg8bit', 'gtFine'),
                                         f.replace('leftImg8bit', 'gtFine').replace('.png', '_instanceIds.png'))))

# 主流程
track_images = [f for f in os.listdir(track_img_dir) if f.endswith('.jpg') or f.endswith('.png')]
idx = 2097
for fname in track_images:
    img_path = os.path.join(track_img_dir, fname)
    bg = cv2.imread(img_path)
    if bg is None:
        continue
    bg_h, bg_w = bg.shape[0:2]
    track_mask = get_track_mask(bg)
    if track_mask is None:
        continue

    track_mask = cv2.resize(track_mask, (1920, 1080))
    paste_points = sample_points_near_track_lr(track_mask, sample_num=1, range_px=40)
    if not paste_points:
        continue
    while True:
        city_img, city_ins = random.choice(city_images)
        person_list = extract_person_instances(city_img, city_ins)
        if person_list:
            break

    bboxes = []
    for i, (px, py) in enumerate(paste_points):
        rgba, _, _ = random.choice(person_list)
        scale = 1
        box = paste_person(bg, rgba, px, py, scale)
        bboxes.append(box)

    
    bg_with_mask = visualize_track_mask(bg, track_mask)
    visual_img = visualize_result(bg_with_mask, bboxes, paste_points)
    visual_img = cv2.resize(visual_img, (960, 540))
    cv2.imshow("result", visual_img)
    cv2.waitKey(1)
    # 保存图像和标签
    out_name = f"{idx:05d}"
    cv2.imwrite(os.path.join(save_img_dir, out_name + ".jpg"), bg)
    with open(os.path.join(save_label_dir, out_name + ".txt"), 'w') as f:
        for cls, cx, cy, w, h in bboxes:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    print(f"✅ 合成 {out_name}.jpg")
    idx += 1
    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": "../images/" + out_name + ".jpg",
        "imageData": None,
        "imageHeight": bg.shape[0],
        "imageWidth": bg.shape[1]
    }

    for cls, cx, cy, w, h in bboxes:
        x_center = cx * bg.shape[1]
        y_center = cy * bg.shape[0]
        box_w = w * bg.shape[1]
        box_h = h * bg.shape[0]
        x1 = x_center - box_w / 2
        y1 = y_center - box_h / 2
        x2 = x_center + box_w / 2
        y2 = y_center + box_h / 2

        shape = {
            "label": "person",
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        labelme_data["shapes"].append(shape)

    # 保存为 .json
    with open(os.path.join(labelme_dir, out_name + ".json"), 'w') as jf:
        json.dump(labelme_data, jf, indent=2)

print(f"🎉 合成完成，共 {idx} 张图像")
