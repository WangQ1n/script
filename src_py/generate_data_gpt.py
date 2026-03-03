import os
import cv2
import numpy as np
import random
import json
from pycocotools.coco import COCO

# -------- CONFIG --------
COCO_JSON = '/home/crrcdt123/datasets/yolov8/datasets/coco/annotations/instances_train2017.json'
COCO_IMG_DIR = '/home/crrcdt123/datasets/yolov8/datasets/coco/images/train2017/'
BACKGROUND_DIR = '/home/crrcdt123/git/script/synthetic_output/'
OUTPUT_IMG_DIR = 'output2/images'
OUTPUT_LABEL_DIR = 'output2/labels'
OUTPUT_LABELME_DIR = 'output2/labelme'
CATEGORIES = ['person', 'train']
PASTE_PER_IMAGE = (1, 3)

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELME_DIR, exist_ok=True)


# -------- AUGMENTATION UTILS --------
def scale_by_y(y, h_img, min_scale=0.3, max_scale=1.0):
    relative_pos = y / h_img
    scale = min_scale + (1 - relative_pos) * (max_scale - min_scale)
    return scale

def feather_alpha(mask, ksize=15):
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), 0)
    return np.clip(blurred / np.max(blurred), 0, 1)

def match_brightness(obj_img, bg_patch):
    mean_obj = np.mean(obj_img[obj_img > 0])
    mean_bg = np.mean(bg_patch[bg_patch > 0])
    ratio = mean_bg / (mean_obj + 1e-5)
    return np.clip(obj_img * ratio, 0, 255).astype(np.uint8)

def motion_blur(img, size=15, angle=0):
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    kernel = kernel / np.sum(kernel)
    return cv2.filter2D(img, -1, kernel)

def apply_random_shadow(image):
    h, w = image.shape[:2]
    top_y = np.random.randint(0, h // 2)
    bottom_y = np.random.randint(h // 2, h)
    shadow_mask = np.zeros_like(image, dtype=np.uint8)
    cv2.rectangle(shadow_mask, (0, top_y), (w, bottom_y), (0, 0, 0), -1)
    alpha = np.random.uniform(0.3, 0.7)
    return cv2.addWeighted(image, 1, shadow_mask, alpha, 0)

def change_color_temperature(img, warm=True):
    lut = np.arange(256, dtype=np.uint8)
    if warm:
        lut = np.clip(lut * 1.1, 0, 255).astype(np.uint8)
    else:
        lut = np.clip(lut * 0.9, 0, 255).astype(np.uint8)
    img[:, :, 2] = cv2.LUT(img[:, :, 2], lut)
    return img

def augment_object(img, mask, scale):
    h, w = mask.shape
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    angle = random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST)
    alpha = feather_alpha(mask)
    if random.random() < 0.5:
        img = motion_blur(img, size=random.randint(5, 15), angle=random.uniform(-30, 30))
    return img, alpha

def paste_object(bg_img, obj_img, alpha, x, y):
    bh, bw = bg_img.shape[:2]
    oh, ow = obj_img.shape[:2]
    bg_patch = bg_img[y:y+oh, x:x+ow]
    obj_img = match_brightness(obj_img, bg_patch)
    roi = bg_img[y:y+oh, x:x+ow]
    blended = (roi * (1 - alpha[..., None]) + obj_img * alpha[..., None]).astype(np.uint8)
    bg_img[y:y+oh, x:x+ow] = blended
    bbox = [0, (x + ow / 2) / bw, (y + oh / 2) / bh, ow / bw, oh / bh]
    return bg_img, bbox

def scale_by_y(y, h_img, min_scale=0.2, max_scale=1.2):
    relative_pos = y / h_img
    scale = min_scale + (relative_pos) * (max_scale - min_scale)
    return scale

def main():
    coco = COCO(COCO_JSON)
    cat_ids = coco.getCatIds(catNms=CATEGORIES)
    img_ids = []
    for id in cat_ids:
      img_ids += (coco.getImgIds(catIds=id))
    anns_by_cat = [aid for aid in img_ids if len(coco.getAnnIds(imgIds=aid, catIds=cat_ids)) > 0]
    bg_files = [f for f in os.listdir(BACKGROUND_DIR) if f.lower().endswith(('jpg', 'png'))]

    for idx, bg_name in enumerate(bg_files):
        bg_path = os.path.join(BACKGROUND_DIR, bg_name)
        bg = cv2.imread(bg_path)
        if bg is None: continue
        if random.random() < 0.5:
            bg = apply_random_shadow(bg)
        if random.random() < 0.3:
            bg = change_color_temperature(bg, warm=random.choice([True, False]))

        bboxes = []
        labelme_polygons = []
        class_names = []
        obj_queue = []

        for _ in range(random.randint(*PASTE_PER_IMAGE)):
            try:
                coco_img_id = random.choice(anns_by_cat)
                ann_ids = coco.getAnnIds(imgIds=coco_img_id, catIds=cat_ids, iscrowd=0)
                anns = coco.loadAnns(ann_ids)
                ann = random.choice(anns)
                seg = coco.annToMask(ann)

                if len(ann['segmentation']) > 1 or np.sum(seg) < 5000: continue
                img_info = coco.loadImgs(coco_img_id)[0]
                coco_img_path = os.path.join(COCO_IMG_DIR, img_info['file_name'])
                obj_img = cv2.imread(coco_img_path)
                if obj_img is None: continue
                bh, bw = bg.shape[:2]
                x, y, w, h = cv2.boundingRect(seg)
                crop = obj_img[y:y+h, x:x+w]
                mask = seg[y:y+h, x:x+w]
                pos_y = np.random.randint(int(bh * 0.1), int(bh * 0.9))
                scale = scale_by_y(pos_y, bh)
                crop, alpha = augment_object(crop, mask, scale)
                oh, ow = crop.shape[:2]
                max_x = bw - ow
                max_y = int(bh * 0.9) - oh
                min_y = int(bh * 0.2)
                if max_x <= 0 or max_y <= min_y:
                    continue
                px = np.random.randint(0, max_x)
                py = np.random.randint(min_y, max_y)
                obj_queue.append((py, crop, alpha, px, py, mask, ann["category_id"]))
            except Exception as e:
                continue

        obj_queue.sort()
        for _, crop, alpha, x, y, mask, class_name in obj_queue:
            bg, bbox = paste_object(bg, crop, alpha, x, y)
            if bbox:
                bboxes.append(bbox)
                class_names.append(coco.loadCats(class_name)[0]["name"])
                # get polygon
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if len(cnt) < 3: continue
                    poly = [(float(pt[0][0] + x), float(pt[0][1] + y)) for pt in cnt]
                    labelme_polygons.append(poly)

        image_out = os.path.join(OUTPUT_IMG_DIR, f"synthetic_{idx:04d}.jpg")
        label_out = os.path.join(OUTPUT_LABEL_DIR, f"synthetic_{idx:04d}.txt")
        labelme_out = os.path.join(OUTPUT_LABELME_DIR, f"synthetic_{idx:04d}.json")
        if (len(bboxes)) == 0:
            continue
        cv2.imwrite(image_out, bg)
        with open(label_out, 'w') as f:
            for idx in range(len(bboxes)):
                f.write("{:s} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(class_names[idx], *bboxes[idx][1:]))
        
        save_labelme_json(labelme_out, bg.shape[:2], bboxes, class_names)

        print(f"Generated: {image_out}, {len(bboxes)} objects")

def save_labelme_json(filename, image_shape, boxes, class_names):
    labelme = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": "../images/" + os.path.basename(filename).replace("json", "jpg"),
        "imageHeight": image_shape[0],
        "imageWidth": image_shape[1],
        "imageData": None,
    }
    for idx in range(len(boxes)):
        cx, cy, w, h = boxes[idx][1:]
        x1 = (cx - w / 2) * image_shape[1]
        y1 = (cy - h / 2) * image_shape[0]
        x2 = (cx + w / 2) * image_shape[1]
        y2 = (cy + h / 2) * image_shape[0]
        polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        labelme['shapes'].append({
            "label": class_names[idx],
            "points": polygon,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })
    with open(filename, 'w') as f:
        json.dump(labelme, f, indent=2)

if __name__ == "__main__":
    main()
