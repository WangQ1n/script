import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import json
import os
import random
import cv2 as cv
import math
from json2mask import readjson, getFileName
import glob
import shutil
from rich.progress import Progress

class GenerateTextureData():
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, label_path, bg_root, forge_root, mode, area_flag):
        super().__init__()
        self.bg_root = bg_root
        self.forge_root = forge_root
        self.train = mode
        self.area_flag = area_flag
        self.shadow = 0
        self.fake = 0
        self.anno_info = dict()
        self.own_anno_info = dict()
        self.cam_rois = dict()
        self._annodir = os.path.join(self.forge_root, "annotations")
        self._imgdir = os.path.join(self.forge_root, "images", "val2017")
        self._annopath = os.path.join(self._annodir, "instances_val2017.json")
        self._imgpath = os.path.join(self._imgdir, "%s.jpg")
        self.bg_roi_path = label_path
        self.own_forge_root = "/home/crrcdt123/datasets2/twoDoor/s8/obstacle_dataset/"
        self.own_forge_label_dir = os.path.join(self.own_forge_root, "labels")
        self.own_forge_img_dir = os.path.join(self.own_forge_root, "images")
        self.bg_start_path = []
        self.bg_end_path = []

        self.load_bg_path()
        self.load_anno()
        self.load_bg_roi()
        self.load_own_forge()

    def load_bg_path(self):
        files = glob.glob(os.path.join(self.bg_root, '*End*.jpg'), recursive=False)
        names = [os.path.basename(file) for file in files]
        times = [int(f"{int(name.split('.')[-2].split('-')[-2] + name.split('.')[-2].split('-')[-1]):.10f}".replace('.', '').ljust(17, '0')[:17]) for name in names]
        start_files = glob.glob(os.path.join(self.bg_root, '*Start*.jpg'), recursive=False)
        start_names = [os.path.basename(file) for file in start_files]
        start_times = [int(f"{int(name.split('.')[-2].split('-')[-2] + name.split('.')[-2].split('-')[-1]):.10f}".replace('.', '').ljust(17, '0')[:17]) for name in start_names]
        sucess_cnt = 0

        for idx, time in enumerate(times):
            tmp_start_idx = []
            tmp_time_dist = []
            tmp_cnt = 0
            for start_idx, start_time in enumerate(start_times):
                if names[idx].split("-")[0] == start_names[start_idx].split("-")[0] and time - start_time > 0 and time - start_time < 3.0e5:  # 1.5min
                    tmp_start_idx.append(start_idx)
                    tmp_time_dist.append(time - start_time)
                    tmp_cnt += 1

            if tmp_cnt > 0:
                if min(tmp_time_dist) < 2.0e5:
                    start_idx = tmp_start_idx[tmp_time_dist.index(min(tmp_time_dist))]
                    self.bg_end_path.append(names[idx])
                    self.bg_start_path.append(start_names[start_idx])
                    sucess_cnt += 1

        print("match sucess:", sucess_cnt, ", false:", len(times) - sucess_cnt)

    def load_anno(self):
        with open(self._annopath, "r") as file:
            self.anno_info = json.load(file)
        self.anno_len = len(self.anno_info["annotations"])

    def load_own_forge(self):
        label_names = getFileName(self.own_forge_label_dir, ".json")
        for name in label_names:
            label_path = os.path.join(self.own_forge_label_dir, name)
            name = name.replace(".json", "")
            polys = []
            polys_list = readjson(label_path)
            for poly in polys_list:
                poly = np.array(poly).astype(np.int32)
                polys.append(poly)
            self.own_anno_info.update({name: polys})
        self.own_anno_len = len(self.own_anno_info)

    def load_bg_roi(self):
        label_names = getFileName(self.bg_roi_path, ".json")
        for name in label_names:
            cam_name = name.split("-")[0]
            # cam_name = name.split("-2024")[0]
            label_path = os.path.join(self.bg_roi_path, name)
            polys = []
            polys_list = readjson(label_path)
            for poly in polys_list:
                poly = np.array(poly).astype(np.int32)
                polys.append(poly)
            self.cam_rois.update({cam_name: polys})

    def fake_shadow(self, img, roi):
        size = img.shape
        shadow_img = img.copy()
        for i in range(2):
            mask = img.copy()
            method = random.randint(0, 2)
            alpha = random.randint(30, 95) / 100.
            blta = 1 - alpha
            if method == 0:
                pts = random.randint(3, 7)
                poly = np.array([[random.randint(1, size[0] - 1),
                                random.randint(1, size[0] - 1)]
                                for i in range(pts)]).astype(np.int32)
                cv.fillPoly(mask, [poly], (0, 0, 0), 1)
                shadow_img = cv.addWeighted(shadow_img, alpha, mask, blta, 0)
            elif method == 1:
                center = np.array([random.randint(1, size[0] - 1),
                                  random.randint(1, size[0] - 1)]).astype(np.int32)
                axes = np.array([random.randint(1, size[0] - 1),
                                random.randint(1, size[0] - 1)]).astype(np.int32)
                angle = random.randint(0, 360)
                startAng, endAng = random.randint(
                    0, 360), random.randint(0, 360)
                cv.ellipse(mask, (int(center[0]), int(center[1])),
                           (int(axes[0]), int(axes[1])), angle, startAng, endAng, (0, 0, 0), -1)
                shadow_img = cv.addWeighted(shadow_img, alpha, mask, blta, 0)
        height, width, _ = img.shape
        roi_img = np.zeros(img.shape, dtype=np.uint8)
        roi_mask = np.zeros([height, width, 1], dtype=np.uint8)
        cv.fillPoly(roi_mask, [roi], (255), 1)
        cv.copyTo(shadow_img, roi_mask, roi_img)
        return roi_img
    
    def get_anno_info(self, idx):
        seg = None
        bbox = None 
        img = None
        instance = self.anno_info["annotations"][idx]
        img_infos = self.anno_info["images"]
        # load img
        img_name = ""
        for img_info in img_infos:
            if instance["image_id"] == img_info["id"]:
                img_name = img_info["file_name"]
        img_path = os.path.join(self._imgdir, img_name)
        img = cv.imread(img_path)
        if img is None:
            print("load img is none:", img_path)
            return seg, bbox, img
        
        if isinstance(instance["segmentation"], dict):
            return seg, bbox, img
        instance_id = np.random.randint(0, len(instance["segmentation"]))
        seg = np.array(instance["segmentation"][instance_id])
        seg = seg.reshape((seg.shape[0]//2, 2))
        bbox = np.array(instance["bbox"])
        return seg, bbox, img

    def get_own_anno_info(self, idx):
        anno_info = self.own_anno_info[idx]
        instance_id = np.random.randint(0, len(anno_info))
        img_path = os.path.join(self.own_forge_img_dir, idx+".jpg")
        img = cv.imread(img_path)
        if img is None:
            print("load img is none:", img_path)
            return
        seg = np.array(anno_info[instance_id])
        # seg = seg.reshape((seg.shape[0]//2, 2))
        bbox = np.array(cv.boundingRect(seg))
        return seg, bbox, img

    def rotate_image_and_contour(self, image, contour, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv.warpAffine(image, rot_mat, (w, h), flags=cv.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        rotated_contour = cv.transform(np.array([contour]), rot_mat)
        return rotated_image, rotated_contour

    def paste_overlay_with_mask(self, background, roi_mask, overlay, mask, position, min_valid_pixels=500):
        # 背景粘贴区域（允许超出）
        rh, rw = overlay.shape[0], overlay.shape[1]
        bh, bw = background.shape[:2]
        x, y = position
        x1_dst, y1_dst = x, y

        # 裁剪范围（确保目标图像区域合法）
        x1_crop = max(0, -x1_dst)
        y1_crop = max(0, -y1_dst)
        x2_crop = min(rw, bw - x1_dst)
        y2_crop = min(rh, bh - y1_dst)

        if x2_crop <= x1_crop or y2_crop <= y1_crop:
            # print("全超出背景区域，跳过")
            return background, False

        # 粘贴
        paste_x1 = max(0, x1_dst)
        paste_y1 = max(0, y1_dst)
        paste_x2 = paste_x1 + (x2_crop - x1_crop)
        paste_y2 = paste_y1 + (y2_crop - y1_crop)

        cropped_patch = overlay[y1_crop:y2_crop, x1_crop:x2_crop]
        cropped_mask = mask[y1_crop:y2_crop, x1_crop:x2_crop]
        bg_roi_mask = roi_mask[paste_y1:paste_y2, paste_x1:paste_x2]
        # 有效像素数量判断
        visible_mask = np.logical_and(cropped_mask > 0, bg_roi_mask > 0)
        visible_pixels = np.sum(visible_mask)
        
        if visible_pixels < min_valid_pixels:
            # print(f"有效像素数量太少：{visible_pixels}，跳过")
            return background, False
        # 粘贴贴图
        bg_roi = background[paste_y1:paste_y2, paste_x1:paste_x2]
        visible_mask_3c = np.repeat(visible_mask[:, :, np.newaxis], 3, axis=2)

        # 粘贴到图像
        bg_roi[visible_mask_3c] = cropped_patch[visible_mask_3c]
        background[paste_y1:paste_y2, paste_x1:paste_x2] = bg_roi
        return background, True

    def mosic(self, bg, roi, texture, seg, bbox):
        height, width, _ = texture.shape
        bg_height, bg_width, _ = bg.shape
        roi_bbox = cv.boundingRect(roi)
        # 缩放 + 旋转
        if roi_bbox[2]*roi_bbox[3] > 1e4:
            roi_type = "large"
            scale = np.random.randint(10, 150)/100.
        else:
            roi_type = "small"
            scale = np.random.randint(10, 150)/100.
        bbox = np.ceil(bbox * scale).astype(np.int32)
        seg = (seg * scale).astype(np.int32)
        texture = cv.resize(texture, (int(width * scale), int(height * scale)))
        while True:
            center = (random.randint(0, bg_width),
                      random.randint(0, bg_height))
            if cv.pointPolygonTest(roi, center, False) >= 0:
                break
        # 旋转
        angle = np.random.randint(0, 360)
        rot_texture, rot_seg = self.rotate_image_and_contour(texture, seg, angle)
        texture_mask = np.zeros((rot_texture.shape[0], rot_texture.shape[1]), dtype=np.uint8)
        cv.fillPoly(texture_mask, [rot_seg], 255)
        roi_mask = np.zeros((bg_height, bg_width), dtype=np.uint8)
        cv.fillPoly(roi_mask, [roi], 255)
        result, success = self.paste_overlay_with_mask(bg.copy(), roi_mask, rot_texture, texture_mask, center, 1000 if roi_type == "large" else 200)
        if success:
            # cv.imshow("result", result)
            # cv.waitKey(0)
            return result
        else:
            return None

    # crop
    def crop(self, idx):
        start_img = None
        roi_img = None
        roi_texture_image = None
        bg_idx = idx
        bg_start_name = self.bg_start_path[bg_idx]
        bg_name = self.bg_end_path[bg_idx]
        self.bg = cv.imread(os.path.join(self.bg_root, bg_name))
        self.bg_start = cv.imread(os.path.join(self.bg_root, bg_start_name))
        if self.bg is None or self.bg_start is None:
            return start_img, roi_img

        height, width, _ = self.bg.shape
        # cam_name = self.bg_path[bg_idx].split("-2024")[0]
        cam_name = self.bg_end_path[bg_idx].split("-")[0]
        if area_flag == "roi0":
            roi_idx = 0
        elif area_flag == "roi1":
            roi_idx = 1
        rois = self.cam_rois[cam_name].copy()
        roi = rois[roi_idx]
        bg_img = self.bg.copy()
        # roi_images = dict()
        # for idx, roi in enumerate(rois):
        roi_bbox = cv.boundingRect(roi)
        roi_img = np.zeros(bg_img.shape, dtype=np.uint8)
        roi_img_start = np.zeros(self.bg_start.shape, dtype=np.uint8)
        roi_mask = np.zeros([height, width, 1], dtype=np.uint8)
        cv.fillPoly(roi_mask, [roi], (255), 1)
        cv.copyTo(bg_img, roi_mask, roi_img)
        cv.copyTo(self.bg_start, roi_mask, roi_img_start)
        roi_img = roi_img[roi_bbox[1]:roi_bbox[1] + roi_bbox[3],
                        roi_bbox[0]:roi_bbox[0] + roi_bbox[2], ...]
        roi_img_start = roi_img_start[roi_bbox[1]:roi_bbox[1] + roi_bbox[3],
                        roi_bbox[0]:roi_bbox[0] + roi_bbox[2], ...]
        # roi_mask = roi_mask[roi_bbox[1]:roi_bbox[1]+roi_bbox[3], roi_bbox[0]:roi_bbox[0]+roi_bbox[2], ...]
        # roi[:, 0] = np.maximum(roi[:, 0] - roi_bbox[0], 0)
        # roi[:, 1] = np.maximum(roi[:, 1] - roi_bbox[1], 0)

            # print("idx:", bg_idx, roi_idx, method)
        tmp_img = roi_img.copy()
        target_size = 224
        if 1:
            roi_img_start = cv.resize(roi_img_start, (target_size, target_size))
            roi_img = cv.resize(roi_img, (target_size, target_size))
            # roi_img = cv.blur(roi_img, (kernel_size, kernel_size))
        else:
            # 计算填充后的图像大小
            roi_texture_image = np.zeros(
                (target_size, target_size, 3), dtype=np.uint8)
            h, w = tmp_img.shape[:2]
            aspect_ratio = w / h
            if aspect_ratio > 1:
                new_w = target_size
                new_h = int(target_size / aspect_ratio)
            else:
                new_h = target_size
                new_w = int(target_size * aspect_ratio)
            # 缩放图像
            resized_image = cv.resize(tmp_img, (new_w, new_h))
            # resized_image = cv.blur(resized_image, (kernel_size, kernel_size))
            # 计算填充区域的位置
            pad_top = (target_size - new_h) // 2
            # pad_bottom = target_size - new_h - pad_top
            pad_left = (target_size - new_w) // 2
            # pad_right = target_size - new_w - pad_left
            # 将图像嵌入到填充后的图像中心
            roi_texture_image[pad_top:pad_top + new_h,
                            pad_left:pad_left + new_w] = resized_image
            roi_img = roi_texture_image

        # roi_images.update({bg_name + "_" + str(idx): roi_img})
        return roi_img_start, roi_img
    
    def getitem(self, index):
        # 随机裁切背景
        start_img = None
        roi_img = None
        roi_texture_image = None
        bg_texture_image = None
        # bg_idx = np.random.randint(0, len(self.bg_end_path))
        bg_idx = index
        bg_start_name = self.bg_start_path[bg_idx]
        bg_name = self.bg_end_path[bg_idx]
        self.bg = cv.imread(os.path.join(self.bg_root, bg_name))
        self.bg_start = cv.imread(os.path.join(self.bg_root, bg_start_name))
        if self.bg is None or self.bg_start is None:
            return start_img, roi_img, roi_texture_image, bg_texture_image, cls, bg_name

        height, width, _ = self.bg.shape
        cam_name = self.bg_end_path[bg_idx].split("-")[0]
        rois = self.cam_rois[cam_name]
        if area_flag == "roi0":
            roi_idx = 0
        elif area_flag == "roi1":
            roi_idx = 1
        elif area_flag == "roi2":
            roi_idx = 2
        else:
            roi_idx = np.random.randint(0, len(rois))
        if roi_idx >= len(rois):
            return start_img, roi_img, roi_texture_image, bg_texture_image, cls, bg_name
        roi = rois[roi_idx].copy()
        roi_bbox = cv.boundingRect(roi)
        # 沿向随机偏移
        vx, vy, x, y = cv.fitLine(roi, cv.DIST_L2, 0, 0.01, 0.01)
        radian = np.arctan2(vy, vx)
        radian2 = np.arctan2(vx, vy)
        offset_dist = np.random.randint(-7, 7)
        roi_offset_x = offset_dist * math.cos(radian)
        roi_offset_y = offset_dist * math.sin(radian)
        offset_dist = np.random.randint(-4, 4)
        roi_offset_x = roi_offset_x+offset_dist * math.cos(radian2)
        roi_offset_y = roi_offset_y+offset_dist * math.sin(radian2)

        # print("拟合线段弧度：", radian, roi_offset_x, roi_offset_y)
        roi[:, 0] = np.minimum(np.maximum(roi[:, 0] + roi_offset_x, 0), width)
        roi[:, 1] = np.minimum(np.maximum(roi[:, 1] + roi_offset_y, 0), height)
        roi_bbox = cv.boundingRect(roi)
        roi_img = np.zeros(self.bg.shape, dtype=np.uint8)
        roi_img_start = np.zeros(self.bg_start.shape, dtype=np.uint8)
        roi_mask = np.zeros([height, width, 1], dtype=np.uint8)
        cv.fillPoly(roi_mask, [roi], (255), 1)
        cv.copyTo(self.bg, roi_mask, roi_img)
        cv.copyTo(self.bg_start, roi_mask, roi_img_start)
        roi_img = roi_img[roi_bbox[1]:roi_bbox[1] + roi_bbox[3],
                          roi_bbox[0]:roi_bbox[0] + roi_bbox[2], ...]
        roi_img_start = roi_img_start[roi_bbox[1]:roi_bbox[1] + roi_bbox[3],
                          roi_bbox[0]:roi_bbox[0] + roi_bbox[2], ...]
        # roi_mask = roi_mask[roi_bbox[1]:roi_bbox[1]+roi_bbox[3], roi_bbox[0]:roi_bbox[0]+roi_bbox[2], ...]
        roi[:, 0] = np.maximum(roi[:, 0] - roi_bbox[0], 0)
        roi[:, 1] = np.maximum(roi[:, 1] - roi_bbox[1], 0)

        roi_texture_img = roi_img.copy()
        draw_cnt = np.random.randint(1, 2)
        while draw_cnt:
            method = np.random.randint(0, 1)
            if method == 0:
                draw_cnt -= 1
                while True:
                    anno_idx = np.random.randint(0, self.anno_len)
                    seg, bbox, texture = self.get_anno_info(anno_idx)
                    if seg is None:
                        continue
                    processed_img = self.mosic(roi_texture_img, roi, texture, seg, bbox)
                    if processed_img is not None:
                        roi_texture_img = processed_img
                        cls = 1
                        break
        
        target_size = 224
        if 1:
            roi_img_start = cv.resize(roi_img_start, (target_size, target_size))
            roi_img = cv.resize(roi_img, (target_size, target_size))
            roi_texture_image = cv.resize(roi_texture_img, (target_size, target_size))
            # kernel_size = np.random.randint(1, 5)
            # roi_texture_image = cv.blur(roi_texture_image, (kernel_size, kernel_size))
            # texture_img = cv.resize(texture_img, (target_size, target_size))
        else:
            # 计算填充后的图像大小
            roi_texture_image = np.zeros(
                (target_size, target_size, 3), dtype=np.uint8)
            h, w = texture_img.shape[:2]
            aspect_ratio = w / h
            if aspect_ratio > 1:
                new_w = target_size
                new_h = int(target_size / aspect_ratio)
            else:
                new_h = target_size
                new_w = int(target_size * aspect_ratio)
            # 缩放图像
            resized_image = cv.resize(texture_img, (new_w, new_h))
            kernel_size = np.random.randint(1, 5)
            # resized_image = cv.blur(resized_image, (kernel_size, kernel_size))
            # 计算填充区域的位置
            pad_top = (target_size - new_h) // 2
            # pad_bottom = target_size - new_h - pad_top
            pad_left = (target_size - new_w) // 2
            # pad_right = target_size - new_w - pad_left
            # 将图像嵌入到填充后的图像中心
            roi_texture_image[pad_top:pad_top + new_h,
                              pad_left:pad_left + new_w] = resized_image
            
            resized_roi_img = cv.resize(roi_img, (new_w, new_h))
            roi_img = np.zeros(
                (target_size, target_size, 3), dtype=np.uint8)
            roi_img[pad_top:pad_top + new_h,
                        pad_left:pad_left + new_w] = resized_roi_img

            resized_roi_img = cv.resize(roi_img_start, (new_w, new_h))
            roi_img_start = np.zeros(
                (target_size, target_size, 3), dtype=np.uint8)
            roi_img_start[pad_top:pad_top + new_h,
                        pad_left:pad_left + new_w] = resized_roi_img
        # 将生成的roi伪数据填充回原图
        bg_texture_image = self.bg.copy()
        src_texture_image = np.zeros(self.bg.shape, dtype=np.uint8)
        src_texture_image[roi_bbox[1]:roi_bbox[1]+roi_bbox[3],
                      roi_bbox[0]:roi_bbox[0]+roi_bbox[2], ...] = roi_texture_img
        cv.copyTo(src_texture_image, roi_mask, bg_texture_image)
        return roi_img_start, roi_img, roi_texture_image, src_texture_image, cls, bg_name

    def __len__(self):
        return len(self.bg_path)


def readAbnormalName(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    # 去除换行符
    abnormal_name = [line.strip() for line in lines]
    return abnormal_name


abnormal_name = readAbnormalName("/home/crrcdt123/git/Siamese-pytorch/abnormal.txt")

forge_root = "/home/crrcdt123/datasets/yolov8/datasets/coco/"
root = "/media/crrcdt123/glam/crrc/data/su8/2door/081706-20250722/"
label_path = "/home/crrcdt123/datasets2/twoDoor/二门列车标签/0817"  # os.path.join(root, "tmp")
bg_path = os.path.join(root, "pictures")
save_path = os.path.join(root, "train")
area_flags = ["roi0", "roi1"]
cnt = 0
uplimit = 3000
if cnt == 1:
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
if 1:
    with Progress() as progress:
        for area_flag in area_flags:
            shadow_ = GenerateTextureData(label_path, bg_path, forge_root, 1, area_flag)
            task = progress.add_task("[cyan]Processing...", total=len(shadow_.bg_end_path))
            for idx in range(0, len(shadow_.bg_end_path)):
                bg_name = shadow_.bg_end_path[idx].split(".")[0]
                # if bg_name not in abnormal_name:
                #     continue
                roi_img_start, roi_img, roi_texture_image, texture_image, result_cls, bg_name = shadow_.getitem(idx)
                if roi_img is None:
                    continue
                # file_dir = bg_name.split('-')[3].split('.')[0]
                file_dir = str(cnt).zfill(5)
                if os.path.isdir(os.path.join(save_path, file_dir)):
                    shutil.rmtree(os.path.join(save_path, file_dir))
                os.mkdir(os.path.join(save_path, file_dir))
                cv.imwrite(os.path.join(save_path, file_dir, "0.jpg"), roi_img_start)
                cv.imwrite(os.path.join(save_path, file_dir, "1.jpg"), roi_img)
                cv.imwrite(os.path.join(save_path, file_dir, "2.jpg"), roi_texture_image)
                cnt += 1
                progress.update(task, advance=1)
else:
    with Progress() as progress:
        for area_flag in area_flags:
            shadow_ = GenerateTextureData(label_path, bg_path, forge_root, 1, area_flag)
            task = progress.add_task("[cyan]Processing...", total=len(shadow_.bg_end_path))
            # while cnt < uplimit:
            for idx in range(0, len(shadow_.bg_end_path)):
                roi_img_start, roi_img = shadow_.crop(idx)
                if roi_img is None:
                    continue
                file_dir = shadow_.bg_end_path[idx].split('.')[0] + "_" + area_flag
                # file_dir = str(cnt).zfill(5)
                if os.path.isdir(os.path.join(save_path, file_dir)):
                    shutil.rmtree(os.path.join(save_path, file_dir))
                os.mkdir(os.path.join(save_path, file_dir))
                cv.imwrite(os.path.join(save_path, file_dir, "0.jpg"), roi_img_start)
                cv.imwrite(os.path.join(save_path, file_dir, "1.jpg"), roi_img)
                cnt += 1
                progress.update(task, advance=1)

# import os
# # 设置目标文件夹的路径
# folder_path = save_path  # 替换为你的文件夹路径
# # 获取文件夹列表
# folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
# # 遍历所有文件夹
# for folder in folders:
#     # 检查文件夹名是否是数字，并进行重命名
#     if folder.isdigit():
#         # 对文件夹名进行格式化，使其长度为5，前面补0
#         new_name = folder.zfill(5)
#         # 获取文件夹的完整路径
#         old_folder_path = os.path.join(folder_path, folder)
#         new_folder_path = os.path.join(folder_path, new_name)
#         # 重命名文件夹
#         if old_folder_path != new_folder_path:
#             os.rename(old_folder_path, new_folder_path)
#             print(f'Renamed "{folder}" to "{new_name}"')
#         else:
#             print(f'Folder "{folder}" is already correctly named.')