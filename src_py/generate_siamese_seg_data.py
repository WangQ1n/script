import os
import glob
import math
import json
import shutil
import random
import cv2 as cv
import numpy as np
from rich.progress import Progress
from json2mask import readjson, getFileName
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


TARGETSIZE = 224
def simple_time_diff(timestamp1, timestamp2):
    """简单版本：直接计算相差毫秒数"""
    
    ts1 = str(timestamp1)
    ts2 = str(timestamp2)
    if ts1[12:14] >= '60':
        ts1 = ts1[:12] + '59' + ts1[14:]
    if ts2[12:14] >= '60':
        ts2 = ts2[:12] + '59' + ts2[14:]
    dt1 = datetime.strptime(ts1[:14], "%Y%m%d%H%M%S")
    dt2 = datetime.strptime(ts2[:14], "%Y%m%d%H%M%S")

    # 添加毫秒部分
    ms1 = int(ts1[14:17])
    ms2 = int(ts2[14:17])
    
    diff_seconds = (dt1 - dt2).total_seconds()
    diff_ms = diff_seconds * 1000 + (ms1 - ms2)
    
    return diff_ms

class GenerateTextureData():
    def __init__(self, label_path, bg_root, forge_root, shuffle, area_flag, anomaly_names=None):
        super().__init__()
        self.bg_root = bg_root
        self.forge_root = forge_root
        self.area_flag = area_flag
        self.shuffle = shuffle
        self.anno_info = dict()
        self.own_anno_info = dict()
        self.cam_rois = dict()
        self.anomaly_names = anomaly_names
        self._annodir = os.path.join(self.forge_root, "annotations")
        self._imgdir = os.path.join(self.forge_root, "images", "train2017")
        self._annopath = os.path.join(self._annodir, "instances_train2017.json")
        self._imgpath = os.path.join(self._imgdir, "%s.jpg")
        self.bg_roi_path = label_path
        self.own_forge_root = "/home/crrcdt123/datasets2/twoDoor/s8/obstacle_dataset/"
        self.own_forge_label_dir = os.path.join(self.own_forge_root, "labels")
        self.own_forge_img_dir = os.path.join(self.own_forge_root, "images")
        self.bg_start_path = []
        self.bg_end_path = []
        self.load_anno()
        self.load_bg_roi()
        self.load_own_forge()
        self.load_bg_path()

    def load_bg_path(self):
        files = glob.glob(os.path.join(
            self.bg_root, '*End*.jpg'), recursive=False)
        files.sort()
        names = [os.path.basename(file) for file in files]
        times = [int(f"{int(name.split('.')[-2].split('-')[-2] + name.split('.')[-2].split('-')[-1]):.10f}".replace(
            '.', '').ljust(17, '0')[:17]) for name in names]
        start_files = glob.glob(os.path.join(
            self.bg_root, '*Start*.jpg'), recursive=False)
        start_files.sort()
        start_names = [os.path.basename(file) for file in start_files]
        start_times = [int(f"{int(name.split('.')[-2].split('-')[-2] + name.split('.')[-2].split('-')[-1]):.10f}".replace(
            '.', '').ljust(17, '0')[:17]) for name in start_names]
        sucess_cnt = 0
        

        start_init_pos = 0
        for idx, time in enumerate(times):
            tmp_start_idx = []
            tmp_time_dist = []
            tmp_cnt = 0
            if self.anomaly_names is not None and names[idx].split(".")[0] not in self.anomaly_names:
                continue
            for start_idx in range(start_init_pos, len(start_times)):
                start_time = start_times[start_idx]
                diff_ms = simple_time_diff(time, start_time)
                if names[idx].split("-")[0] == start_names[start_idx].split("-")[0] and diff_ms > 0 and diff_ms < 3.0e5:  # 300s
                    tmp_start_idx.append(start_idx)
                    tmp_time_dist.append(diff_ms)
                    tmp_cnt += 1
                elif tmp_cnt > 0:
                    break
            # 首站发车时存在停车过长现象
            if tmp_cnt > 0 and min(tmp_time_dist) < 1e5:
                start_idx = tmp_start_idx[tmp_time_dist.index(
                    min(tmp_time_dist))]
                self.bg_end_path.append(names[idx])
                self.bg_start_path.append(start_names[start_idx])
                sucess_cnt += 1
                start_init_pos = start_idx # max(start_idx - 10, 0)
            else:
                print(names[idx], time)
        if self.shuffle:
            indices = list(range(len(self.bg_end_path)))
            random.shuffle(indices)

            # 根据打乱后的索引重新排列两个列表
            self.bg_start_path = [self.bg_start_path[i] for i in indices]
            self.bg_end_path = [self.bg_end_path[i] for i in indices]
        print(self.bg_root, " match sucess:", sucess_cnt,
              ", false:", len(times) - sucess_cnt)

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
        bbox = np.array(instance["bbox"]).astype(np.int16)
        seg = np.array(instance["segmentation"][instance_id])
        seg = seg.reshape((seg.shape[0]//2, 2))
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

    def mask_to_yolo(self, mask, img_width, img_height, offset_x, offset_y):
        mask_int8 = (mask.astype(np.uint8) * 255).astype(np.uint8)
        contours, _ = cv.findContours(
            mask_int8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # cv.imshow("vis", mask_int8)
        # cv.waitKey(0)
        yolo_labels = []
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            # 转YOLO归一化格式
            # x_center = (offset_x + x + w / 2) / img_width
            # y_center = (offset_y + y + h / 2) / img_height
            # w_norm = w / img_width
            # h_norm = h / img_height
            # yolo_labels.append(f"{0} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            # 归一化坐标
            coords = []
            for p in contour.squeeze(1):  # (N,1,2) -> (N,2)
                x, y = p
                coords.append((offset_x + x) / img_width)
                coords.append((offset_y + y) / img_height)

            # YOLO 格式：class_id x1 y1 x2 y2 ...
            label_line = f"{0} " + " ".join([f"{c:.6f}" for c in coords])
            yolo_labels.append(label_line)

        return yolo_labels

    def rotate_image_and_contour(self, image, contour, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv.warpAffine(
            image, rot_mat, (w, h), flags=cv.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        rotated_contour = cv.transform(np.array([contour]), rot_mat)
        return rotated_image, rotated_contour

    def paste_overlay_with_mask(self, background, roi_mask, overlay, mask, position, min_valid_pixels=500):
        label = []
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
            return background, False, label

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
            return background, False, label
        # 粘贴贴图
        bg_roi = background[paste_y1:paste_y2, paste_x1:paste_x2]
        visible_mask_3c = np.repeat(visible_mask[:, :, np.newaxis], 3, axis=2)
        label = self.mask_to_yolo(visible_mask, bw, bh, paste_x1, paste_y1)
        # 粘贴到图像
        bg_roi[visible_mask_3c] = cropped_patch[visible_mask_3c]
        background[paste_y1:paste_y2, paste_x1:paste_x2] = bg_roi
        return background, True, label

    def mosic(self, bg_img, roi, texture_img, texture_contour):
        texture_height, texture_width, _ = texture_img.shape
        bg_height, bg_width, _ = bg_img.shape
        roi_bbox = cv.boundingRect(roi)
        # 缩放
        if roi_bbox[2]*roi_bbox[3] > 1e4:
            roi_type = "large"
            scale = np.random.randint(30, 150)/100.
        else:
            roi_type = "small"
            scale = np.random.randint(30, 150)/100.
        texture_contour = (texture_contour * scale).astype(np.int32)
        texture_img = cv.resize(
            texture_img, (int(texture_width * scale), int(texture_height * scale)))
        while True:
            center = (random.randint(0, bg_width),
                      random.randint(0, bg_height))
            if cv.pointPolygonTest(roi, center, False) >= 0:
                break
        # 旋转
        angle = np.random.randint(0, 360)
        rot_texture, rot_texture_contour = self.rotate_image_and_contour(
            texture_img, texture_contour, angle)
        texture_mask = np.zeros(
            (rot_texture.shape[0], rot_texture.shape[1]), dtype=np.uint8)
        cv.fillPoly(texture_mask, [rot_texture_contour], 255)
        roi_mask = np.zeros((bg_height, bg_width), dtype=np.uint8)
        cv.fillPoly(roi_mask, [roi], 255)
        result, success, label = self.paste_overlay_with_mask(bg_img.copy(
        ), roi_mask, rot_texture, texture_mask, center, 1000 if roi_type == "large" else 200)
        if success:
            # cv.imshow("result", result)
            # cv.waitKey(0)
            return result, label
        else:
            return None, label

    def random_offset(self, roi, dist_x, dist_y):
        vx, vy, x, y = cv.fitLine(roi, cv.DIST_L2, 0, 0.01, 0.01)
        radian = np.arctan2(vy, vx)
        radian2 = np.arctan2(vx, vy)
        offset_dist = np.random.randint(-dist_x, dist_x)
        roi_offset_x = offset_dist * math.cos(radian)
        roi_offset_y = offset_dist * math.sin(radian)
        offset_dist = np.random.randint(-dist_y, dist_y)
        roi_offset_x = roi_offset_x+offset_dist * math.cos(radian2)
        roi_offset_y = roi_offset_y+offset_dist * math.sin(radian2)
        return roi_offset_x, roi_offset_y

    def resize_with_padding(self, img, size):
        h, w = img.shape[:2]
        aspect_ratio = w / h
        if aspect_ratio > 1:
            new_w, new_h = size, int(size / aspect_ratio)
        else:
            new_h, new_w = size, int(size * aspect_ratio)

        pad_top = (size - new_h) // 2
        pad_left = (size - new_w) // 2

        resized = cv.resize(img, (new_w, new_h))
        out = np.zeros((size, size, 3), dtype=np.uint8)
        out[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
        return out

    def crop(self, bg_idx, keep_aspect=False):
        bg_start_name = self.bg_start_path[bg_idx]
        bg_name = self.bg_end_path[bg_idx]
        bg_end = cv.imread(os.path.join(self.bg_root, bg_name))
        bg_start = cv.imread(os.path.join(self.bg_root, bg_start_name))
        if bg_end is None or bg_start is None:
            return None, None, bg_name

        height, width, _ = bg_end.shape
        cam_name = self.bg_end_path[bg_idx].split("-")[0]
        roi_idx = 0 if self.area_flag == "roi0" else 1
        roi = self.cam_rois[cam_name][roi_idx]
        bg_img = bg_end.copy()
        roi_bbox = cv.boundingRect(roi)
        roi_mask = np.zeros([height, width, 1], dtype=np.uint8)
        cv.fillPoly(roi_mask, [roi], (255), 1)
        roi_end_img = np.zeros(bg_img.shape, dtype=np.uint8)
        roi_start_img = np.zeros(bg_start.shape, dtype=np.uint8)
        cv.copyTo(bg_img, roi_mask, roi_end_img)
        cv.copyTo(bg_start, roi_mask, roi_start_img)
        x, y, w, h = roi_bbox
        roi_end_img = roi_end_img[y:y+h, x:x+w]
        roi_start_img = roi_start_img[y:y+h, x:x+w]
        if keep_aspect is False:
            roi_start_img = cv.resize(
                roi_start_img, (TARGETSIZE, TARGETSIZE))
            roi_end_img = cv.resize(roi_end_img, (TARGETSIZE, TARGETSIZE))
        else:
            roi_end_img = self.resize_with_padding(roi_end_img, TARGETSIZE)
            roi_start_img = self.resize_with_padding(roi_start_img, TARGETSIZE)
        return roi_start_img, roi_end_img, bg_name

    def get_roi_with_texture(self, index,
                keep_aspect: bool = False,
                max_offset_x: int = 4,
                max_offset_y: int = 2):
        """
        获取带伪造纹理的ROI样本。

        Args:
            index (int): 背景图片索引。
            target_size (int): 输出图像大小（方形）。
            keep_aspect (bool): 是否保持长宽比缩放+padding。
            max_offset_x (int): ROI随机x偏移范围。
            max_offset_y (int): ROI随机y偏移范围。

        Returns:
            tuple: (roi_start_img, roi_end_img, roi_texture_image, bg_texture_image, labels, bg_end_name)
        """
        roi_texture_image = None
        bg_texture_image = None
        labels = []
        bg_start_name = self.bg_start_path[index]
        bg_end_name = self.bg_end_path[index]
        bg_end_img = cv.imread(os.path.join(self.bg_root, bg_end_name))
        bg_start_img = cv.imread(os.path.join(self.bg_root, bg_start_name))
        if bg_end_img is None or bg_start_img is None:
            return None, None, None, None, [], bg_end_name

        height, width, _ = bg_end_img.shape
        cam_name = self.bg_end_path[index].split("-")[0]
        rois = self.cam_rois[cam_name]
        if self.area_flag == "roi0":
            roi_idx = 0
        elif self.area_flag == "roi1":
            roi_idx = 1
        else:
            roi_idx = np.random.randint(0, len(rois))
        roi = rois[roi_idx].copy()
        # 随机偏移ROI
        offset_x, offset_y = self.random_offset(roi, max_offset_x, max_offset_y)
        roi[:, 0] = np.clip(roi[:, 0] + offset_x, 0, width)
        roi[:, 1] = np.clip(roi[:, 1] + offset_y, 0, height)
        # ROI mask
        roi_bbox = cv.boundingRect(roi)
        x, y, w, h = roi_bbox
        roi_mask = np.zeros((height, width, 1), dtype=np.uint8)
        cv.fillPoly(roi_mask, [roi], 255, 1)
        roi_bbox = cv.boundingRect(roi)
        roi_mask = np.zeros([height, width, 1], dtype=np.uint8)
        cv.fillPoly(roi_mask, [roi], (255), 1)
    
        roi_end_img = np.zeros(bg_end_img.shape, dtype=np.uint8)
        roi_start_img = np.zeros(bg_start_img.shape, dtype=np.uint8)
        cv.copyTo(bg_end_img, roi_mask, roi_end_img)
        cv.copyTo(bg_start_img, roi_mask, roi_start_img)

        roi_end_img = roi_end_img[y:y+h, x:x+w]
        roi_start_img = roi_start_img[y:y+h, x:x+w]
        # 调整ROI坐标到局部框
        roi[:, 0] = np.maximum(roi[:, 0] - roi_bbox[0], 0)
        roi[:, 1] = np.maximum(roi[:, 1] - roi_bbox[1], 0)
        # --- 伪造纹理 ---
        roi_texture_img = roi_end_img.copy()
        draw_cnt = np.random.randint(1, 2)
        while draw_cnt:
            anno_idx = np.random.randint(0, self.anno_len)
            seg, _, texture = self.get_anno_info(anno_idx)
            if seg is None:
                continue
            processed_img, label = self.mosic(
                roi_texture_img, roi, texture, seg)
            if processed_img is not None:
                roi_texture_img = processed_img
                labels.append(label)
                draw_cnt -= 1
        if keep_aspect:
            roi_start_img = self.resize_with_padding(roi_start_img, TARGETSIZE)
            roi_end_img = self.resize_with_padding(roi_end_img, TARGETSIZE)
            roi_texture_image = self.resize_with_padding(roi_texture_img, TARGETSIZE)
        else:
            roi_start_img = cv.resize(roi_start_img, (TARGETSIZE, TARGETSIZE))
            roi_end_img = cv.resize(roi_end_img, (TARGETSIZE, TARGETSIZE))
            roi_texture_image = cv.resize(roi_texture_img, (TARGETSIZE, TARGETSIZE))

        # --- 将ROI伪数据填充回原图 ---
        bg_texture_image = bg_end_img.copy()
        src_texture_image = np.zeros_like(bg_end_img)
        src_texture_image[y:y+h, x:x+w] = roi_texture_img
        cv.copyTo(src_texture_image, roi_mask, bg_texture_image)
        return roi_start_img, roi_end_img, roi_texture_image, src_texture_image, labels, bg_end_name

    def __len__(self):
        return len(self.bg_end_path)


def readAbnormalName(path, mode=0):
    with open(path, 'r') as f:
        lines = f.readlines()
    # 去除换行符
    if mode == 0:
        abnormal_name = [line.strip().split("-")[0] for line in lines]
    elif mode == 1:
        abnormal_name = [line.strip().replace("_roi0.jpg", "").replace("_roi1.jpg", "") for line in lines]
    else:
        abnormal_name = [line.strip().split(".")[0] for line in lines]
    return abnormal_name


sub_dirs = ["0818"]
data_dirs = ["hard_samples"]
for dir in sub_dirs:
    forge_root = "/home/crrcdt123/datasets/yolov8/datasets/coco/"
    root = os.path.join("/media/crrcdt123/glam/crrc/data/su8/2door/", dir)
    label_path = "/home/crrcdt123/datasets2/twoDoor/二门列车标签/" + dir[:4]
    # label_path = "/media/crrcdt123/glam/crrc/data/su8/2door/unknow/images_labels"
    base_bg_path = os.path.join(root, "clear")
    base_save_path = os.path.join(root, "clear")
    for datatime in data_dirs:
        bg_path = os.path.join(base_bg_path, datatime)
        save_path = os.path.join(base_save_path, "yolo2_" + datatime)
        if not os.path.exists(bg_path):
            continue
        prefix = ''  # 误检图像前缀 FP, TP, TN, FN
        area_flags = ["roi0", "roi1"]  # "roi0", "roi1"
        mode = 1    # 0--mask, 1--crop
        is_random = False
        is_abnormal = False
        if is_abnormal:
            abnormal_name = readAbnormalName(
                "./anomaly.txt", 1)
            full_abnormal_name = readAbnormalName(
                "./anomaly.txt", 2)
        else:
            abnormal_name = None
        generator = GenerateTextureData(
            label_path, bg_path, forge_root, shuffle=is_random, area_flag=area_flags[0], anomaly_names=abnormal_name)

        def CropWorker(idx, area_flag):
            roi_img_start, roi_img, name = generator.crop(idx)
            if roi_img is None:
                print("roi img is None")
                return False
            file_name = name.split('.')[0] + "_" + area_flag
            if is_abnormal and file_name not in full_abnormal_name:
                return False
            if prefix != '':
                file_name = prefix + "_" + file_name
            cv.imwrite(os.path.join(save_path, "test",
                    file_name + ".jpg"), roi_img)
            cv.imwrite(os.path.join(save_path, "test2",
                                    file_name + ".jpg"), roi_img_start)
            return True

        def MosicWorker(idx, area_flag):
            bg_name = generator.bg_end_path[idx].split(".")[0]
            if is_abnormal and bg_name not in abnormal_name:
                return False
            roi_img_start, roi_img, roi_texture_image, _, labels, bg_name = generator.get_roi_with_texture(
                idx)
            if roi_img is None:
                return False
            file_name = bg_name.split(".")[0] + "_" + area_flag
            if prefix != '':
                file_name = prefix + "_" + file_name
            cv.imwrite(os.path.join(save_path, "train2",
                                    file_name + ".jpg"), roi_img_start)
            cv.imwrite(os.path.join(save_path, "train3",
                    file_name + ".jpg"), roi_img)
            cv.imwrite(os.path.join(save_path, "train",
                                    file_name + ".jpg"), roi_texture_image)
            with open(os.path.join(save_path, "labels", file_name + ".txt", ), "w") as f:
                for label in labels:
                    f.write("\n".join(label))
            return True

        cnt = 0
        uplimit = min(1000, len(generator.bg_end_path)) if is_random else len(generator.bg_end_path)
            
        if mode == 0:
            if cnt == 0:
                if os.path.isdir(save_path):
                    shutil.rmtree(save_path)
                os.mkdir(save_path)
                os.mkdir(os.path.join(save_path, "train"))
                os.mkdir(os.path.join(save_path, "train2"))
                os.mkdir(os.path.join(save_path, "train3"))
                os.mkdir(os.path.join(save_path, "labels"))
            with Progress() as progress:
                for area_flag in area_flags:
                    generator.area_flag = area_flag
                    task = progress.add_task(
                        "[cyan]Processing images...", total=uplimit)
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        futures = [executor.submit(MosicWorker, f, area_flag)
                                for f in range(uplimit)]
                        for future in as_completed(futures):
                            if future.result():
                                progress.update(task, advance=1)
        elif mode == 1:
            if cnt == 0:
                if os.path.isdir(save_path):
                    shutil.rmtree(save_path)
                os.mkdir(save_path)
                os.mkdir(os.path.join(save_path, "test"))
                os.mkdir(os.path.join(save_path, "test2"))
            with Progress() as progress:
                for area_flag in area_flags:
                    generator.area_flag = area_flag
                    task = progress.add_task(
                        "[cyan]Processing images...", total=len(generator.bg_end_path))
                    with ThreadPoolExecutor(max_workers=12) as executor:
                        futures = [executor.submit(CropWorker, f, area_flag)
                                for f in range(len(generator.bg_end_path))]
                        for future in as_completed(futures):
                            if future.result():
                                progress.update(task, advance=1)
