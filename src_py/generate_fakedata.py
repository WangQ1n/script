import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import json
import os
import random
import cv2 as cv
import torch
import math
from torchvision import transforms
from json2mask import readjson, getFileName


class GenerateTextureData2(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, bg_root, forge_root, mode, gen_size):
        super().__init__()
        self.bg_root = bg_root
        self.forge_root = forge_root
        self.train = mode
        self.gen_size = gen_size
        self.shadow = 0
        self.fake = 0
        self.anno_info = dict()
        self.cam_rois = dict()
        if self.train:
            self._annodir = os.path.join(self.forge_root, "annotations")
            self._imgdir = os.path.join(self.forge_root, "images", "val2017")
            self._annopath = os.path.join(
                self._annodir, "instances_val2017.json")
            self._imgpath = os.path.join(self._imgdir, "%s.jpg")
            self.bg_roi_path = os.path.join(self.bg_root, "roi")
            # self.ids = [name.replace(".xml", "") for name in os.listdir(self._annodir)]
            # ori_length = len(self.ids)
            # for idx, id in enumerate(self.ids):
            #     if os.path.exists(os.path.join(self._annopath % id)) and \
            #       os.path.exists(os.path.join(self._imgpath % id)):
            #         tree = ET.parse(self._annopath % id)
            #         objs = tree.findall('object')
            #         if len(objs) == 0:
            #             print("del------")
            #             del self.ids[idx]
            #     else:
            #         print("del------")
            #         del self.ids[idx]
            # del self.ids[0:1000]
            # now_length = len(self.ids)
            # print("length: %d --> %d" % (ori_length, now_length))
            self.bg_name = "good"
        else:
            self.bg_name = "abnormal"
        self.load_anno()
        self.load_bg_roi()
        self.bg_path = os.listdir(os.path.join(self.bg_root))

    def load_anno(self):
        with open(self._annopath, "r") as file:
            self.anno_info = json.load(file)
        self.anno_len = len(self.anno_info["annotations"])

    def load_bg_roi(self):
        label_names = getFileName(self.bg_roi_path, ".json")

        for name in label_names:
            cam_name = name.split("-")[0]
            label_path = os.path.join(self.bg_roi_path, name)
            # img_path = label_path.replace(".json", ".jpg")
            # if not os.path.exists(img_path):
            #     print("图像文件缺失：%s" % img_path)
            #     continue
            # img = cv.imread(img_path)
            # mask = np.zeros(img.shape, dtype=np.uint8)
            polys = []
            polys_list = readjson(label_path)
            for poly in polys_list:
                poly = np.array(poly).astype(np.int32)
                polys.append(poly)
                # cv.fillPoly(mask, [poly], (0, 255, 0), 1)
            self.cam_rois.update({cam_name: polys})
            # cv.imshow("img", mask)
            # cv.waitKey(0)

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

    def mosic(self, bg, idx, roi):
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
            return
        height, width, _ = img.shape
        if isinstance(instance["segmentation"], dict):
            return None
        instance_id = np.random.randint(0, len(instance["segmentation"]))
        seg = np.array(instance["segmentation"][instance_id])
        seg = seg.reshape((seg.shape[0]//2, 2))
        bbox = np.array(instance["bbox"])
        roi_bbox = cv.boundingRect(roi)
        area = cv.contourArea(roi)
        if roi_bbox[2]*roi_bbox[3] > 1e4:
            roi_type = "large"
            scale = np.random.randint(10, 100)/100.
        else:
            roi_type = "small"
            scale = np.random.randint(10, 100)/100.
        seg = seg * scale
        seg = seg.astype(np.int32)
        bbox = np.ceil(bbox * scale)
        bbox = bbox.astype(np.int32)
        img = cv.resize(img, (int(width * scale), int(height * scale)))
        height, width, _ = img.shape
        bg_height, bg_width, _ = bg.shape
        # 在掩模上绘制选定的轮廓
        seg_mask = np.zeros([height, width, 1], dtype=np.uint8)
        cv.drawContours(seg_mask, [seg], -1, (255), thickness=cv.FILLED)
        while True:
            center = (random.randint(0, bg_width),
                      random.randint(0, bg_height))
            if cv.pointPolygonTest(roi, center, False) >= 0:
                break
        pix_cnt = 0
        roi_img = np.zeros(bg.shape, dtype=np.uint8)
        roi_mask = np.zeros([bg_height, bg_width, 1], dtype=np.uint8)
        cv.fillPoly(roi_mask, [roi], (255), 1)
        cv.copyTo(bg, roi_mask, roi_img)
        for h in range(bbox[3]):
            for w in range(bbox[2]):
                if bbox[1]+h >= height or bbox[0]+w >= width:
                    continue
                if cv.pointPolygonTest(seg, [int(bbox[0]+w), int(bbox[1]+h)], False) < 0:
                    continue
                if cv.pointPolygonTest(roi, [center[0]+w, center[1]+h], False) >= 0:
                    roi_img[center[1]+h, center[0] +
                            w] = img[bbox[1]+h, bbox[0]+w]
                    pix_cnt = pix_cnt+1
        if (roi_type == "large" and pix_cnt > area/10.) or (roi_type == "small" and pix_cnt > area/5.):
            return roi_img
        else:
            return None
        # center = [np.random.randint(0, bg.shape[1]),
        #           np.random.randint(0, bg.shape[0])]  # [x, y]
        # x_start = int(max(0, center[0] - bbox[2]//2))
        # y_start = int(max(0, center[1] - bbox[3]//2))
        # x_end = int(min(bg_width, center[0] + bbox[2]//2))
        # y_end = int(min(bg_height, center[1] + bbox[3]//2))
        # seg_x_start = int(bbox[0] if x_start > 0 else bbox[0] + bbox[2]//2 - center[0])
        # seg_y_start = int(bbox[1] if y_start > 0 else bbox[1] + bbox[3]//2 - center[1])
        # seg_x_end = seg_x_start + x_end - x_start
        # seg_y_end = seg_y_start + y_end - y_start
        # if (x_end-x_start) * (y_end- y_start) > 200:
        #     print(y_end - y_start, x_end - x_start, seg_y_end-seg_y_start, seg_x_end-seg_x_start)
        #     cv.copyTo(img[seg_y_start:seg_y_end, seg_x_start:seg_x_end], seg_mask[seg_y_start:seg_y_end, seg_x_start:seg_x_end],
        #               bg[y_start:y_end, x_start:x_end])
        #     result_img = np.zeros(bg.shape, dtype=np.uint8)
        #     cv.copyTo(bg, bg_mask, result_img)
            # cv.rectangle(img, [seg_x_start, seg_y_start], [seg_x_end, seg_y_end], (0, 255, 0), 1)
            # cv.rectangle(bg, [x_start, y_start], [x_end, y_end], (0, 255, 0), 1)
            # cv.imshow("img", img)
            # cv.imshow("fuse", bg)
            # cv.imshow("seg_mask", seg_mask)
            # cv.waitKey(0)
        #     return result_img
        # else:
        #     return None

    def __getitem__(self, index):
        # 随机裁切背景
        bg_idx = np.random.randint(0, len(self.bg_path))
        # bg_idx = 38
        self.bg = cv.imread(os.path.join(self.bg_root, self.bg_path[bg_idx]))
        if self.bg is None:
            return (np.array([0]), np.array([0])), -1

        height, width, _ = self.bg.shape
        cam_name = self.bg_path[bg_idx].split("-")[0]
        rois = self.cam_rois[cam_name]
        roi_idx = np.random.randint(0, len(rois))
        # roi_idx = 1

        roi = rois[roi_idx].copy()
        bg_img = self.bg.copy()
        # center_x, center_y, width, height -> x, y, w, h
        # roi = [max(0, roi[0]-roi[2]//2), max(0, roi[1]-roi[3]//2), roi[2], roi[3]]
        # if roi_idx == 0:
        #     offset_x = np.random.randint(-15, 15)
        #     offset_y = np.random.randint(-15, 15)
        #     offset_w = np.random.randint(-5, 5)
        #     offset_h = np.random.randint(-3, 3)
        # else:
        #     offset_x = np.random.randint(-3, 3)
        #     offset_y = np.random.randint(-3, 3)
        #     offset_w = np.random.randint(-1, 1)
        #     offset_h = np.random.randint(-1, 1)
        # roi = [ max(0, min(width, roi[0] + offset_x)),
        #         max(0, min(height, roi[1] + offset_y)),
        #         max(0, roi[2] + offset_w),
        #         max(0, roi[3] + offset_h)]
        # bg_img = bg_img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], ...]
        roi_bbox = cv.boundingRect(roi)
        print(roi_bbox)
        vx, vy, x, y = cv.fitLine(roi, cv.DIST_L2, 0, 0.01, 0.01)
        # 沿向
        radian = np.arctan2(vy, vx)
        radian2 = np.arctan2(vx, vy)
        # resize rotate
        if roi_bbox[2]*roi_bbox[3] > 1e4:
            roi_type = "large"
            offset_dist = np.random.randint(-50, 50)
            roi_offset_x = offset_dist * math.cos(radian)
            roi_offset_y = offset_dist * math.sin(radian)
            offset_dist = np.random.randint(-10, 10)
            roi_offset_x = roi_offset_x+offset_dist * math.cos(radian2)
            roi_offset_y = roi_offset_y+offset_dist * math.sin(radian2)
        else:
            roi_type = "small"
            offset_dist = np.random.randint(-7, 7)
            roi_offset_x = offset_dist * math.cos(radian)
            roi_offset_y = offset_dist * math.sin(radian)
            offset_dist = np.random.randint(-3, 3)
            roi_offset_x = roi_offset_x+offset_dist * math.cos(radian2)
            roi_offset_y = roi_offset_y+offset_dist * math.sin(radian2)
        # 纵向

        # print("拟合线段弧度：", radian, roi_offset_x, roi_offset_y)
        roi[:, 0] = np.minimum(np.maximum(roi[:, 0] + roi_offset_x, 0), width)
        roi[:, 1] = np.minimum(np.maximum(roi[:, 1] + roi_offset_y, 0), height)
        roi_bbox = cv.boundingRect(roi)
        # print(roi_bbox)

        roi_img = np.zeros(bg_img.shape, dtype=np.uint8)
        roi_mask = np.zeros([height, width, 1], dtype=np.uint8)
        cv.fillPoly(roi_mask, [roi], (255), 1)
        cv.copyTo(bg_img, roi_mask, roi_img)
        roi_img = roi_img[roi_bbox[1]:roi_bbox[1]+roi_bbox[3],
                          roi_bbox[0]:roi_bbox[0]+roi_bbox[2], ...]
        # roi_mask = roi_mask[roi_bbox[1]:roi_bbox[1]+roi_bbox[3], roi_bbox[0]:roi_bbox[0]+roi_bbox[2], ...]
        roi[:, 0] = np.maximum(roi[:, 0] - roi_bbox[0], 0)
        roi[:, 1] = np.maximum(roi[:, 1] - roi_bbox[1], 0)
        # cv.imshow("roi", roi_img)
        # cv.rectangle(bg_img, [roi_bbox[0], roi_bbox[1]], [roi_bbox[0] + roi_bbox[2], roi_bbox[1]+roi_bbox[3]], (0, 255, 0), 1)
        # cv.imshow("bg", bg_img)
        # cv.waitKey(0)
        # # print(bg_img.shape, roi, bg_idx, roi_idx)
        if self.train:
            target = np.random.randint(0, 4)
            # target = 2
            # print("idx:", bg_idx, roi_idx, target)
            tmp_img = roi_img.copy()
            if target == 0:
                processed_img = self.fake_shadow(tmp_img, roi)
            elif target == 1:
                while True:
                    anno_idx = np.random.randint(0, self.anno_len)
                    processed_img = self.mosic(tmp_img, anno_idx, roi)
                    if processed_img is not None:
                        break
            elif target == 2:
                target = 1
                tmp_img = self.fake_shadow(tmp_img, roi)
                while True:
                    anno_idx = np.random.randint(0, self.anno_len)
                    processed_img = self.mosic(tmp_img, anno_idx, roi)
                    if processed_img is not None:
                        break
            elif target == 3:
                target = 1
                while True:
                    anno_idx = np.random.randint(0, self.anno_len)
                    processed_img = self.mosic(tmp_img, anno_idx, roi)
                    if processed_img is not None:
                        break
                processed_img = self.fake_shadow(processed_img, roi)
        else:
            tmp_img = bg_img.copy()
            processed_img = tmp_img.copy()
            target = 1

        # 计算填充后的图像大小
        # target_size = 224
        # padded_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        # h, w = processed_img.shape[:2]
        # aspect_ratio = w / h
        # if aspect_ratio > 1:
        #     new_w = target_size
        #     new_h = int(target_size / aspect_ratio)
        # else:
        #     new_h = target_size
        #     new_w = int(target_size * aspect_ratio)

        # # 缩放图像
        # resized_image = cv.resize(processed_img, (new_w, new_h))

        # # 计算填充区域的位置
        # pad_top = (target_size - new_h) // 2
        # pad_bottom = target_size - new_h - pad_top
        # pad_left = (target_size - new_w) // 2
        # pad_right = target_size - new_w - pad_left

        # # 将图像嵌入到填充后的图像中心
        # padded_image[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_image
        padded_image = cv.resize(processed_img, (224, 224))
        # roi_img = np.zeros(bg_img.shape, dtype=np.uint8)
        # roi_img[roi_bbox[1]:roi_bbox[1]+roi_bbox[3], roi_bbox[0]:roi_bbox[0]+roi_bbox[2], ...] = processed_img
        # cv.copyTo(roi_img, roi_mask, bg_img)
        # cv.imshow('Resized Image', bg_img)
        # cv.waitKey(0)
        # self.fake = self.fake + 1
        # print(self.fake)
        # save_path = "/home/crrcdt123/datasets2/二门防夹数据/s8-20240307/all/" + self.bg_path[bg_idx].replace(".jpg", "_") + str(self.fake) + ".jpg"
        # cv.imwrite(save_path, bg_img)
        return (roi_img, padded_image), target

    def __len__(self):
        if self.train:
            return 99999
        else:
            return 100


class GenerateTextureData():
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, bg_root, forge_root, mode):
        super().__init__()
        self.bg_root = bg_root
        self.forge_root = forge_root
        self.train = mode
        self.shadow = 0
        self.fake = 0
        self.anno_info = dict()
        self.cam_rois = dict()

        self._annodir = os.path.join(self.forge_root, "annotations")
        self._imgdir = os.path.join(self.forge_root, "images", "val2017")
        self._annopath = os.path.join(self._annodir, "instances_val2017.json")
        self._imgpath = os.path.join(self._imgdir, "%s.jpg")
        self.bg_roi_path = os.path.join(self.bg_root, "roi")

        self.load_anno()
        self.load_bg_roi()
        self.bg_path = os.listdir(os.path.join(self.bg_root))

    def load_anno(self):
        with open(self._annopath, "r") as file:
            self.anno_info = json.load(file)
        self.anno_len = len(self.anno_info["annotations"])

    def load_bg_roi(self):
        label_names = getFileName(self.bg_roi_path, ".json")
        for name in label_names:
            cam_name = name.split("-")[0]
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

    def mosic(self, bg, idx, roi):
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
            return
        height, width, _ = img.shape
        if isinstance(instance["segmentation"], dict):
            return None
        instance_id = np.random.randint(0, len(instance["segmentation"]))
        seg = np.array(instance["segmentation"][instance_id])
        seg = seg.reshape((seg.shape[0]//2, 2))
        bbox = np.array(instance["bbox"])
        roi_bbox = cv.boundingRect(roi)
        area = cv.contourArea(roi)
        if roi_bbox[2]*roi_bbox[3] > 1e4:
            roi_type = "large"
            scale = np.random.randint(10, 100)/100.
        else:
            roi_type = "small"
            scale = np.random.randint(10, 100)/100.
        seg = seg * scale
        seg = seg.astype(np.int32)
        bbox = np.ceil(bbox * scale)
        bbox = bbox.astype(np.int32)
        img = cv.resize(img, (int(width * scale), int(height * scale)))
        height, width, _ = img.shape
        bg_height, bg_width, _ = bg.shape
        # 在掩模上绘制选定的轮廓
        seg_mask = np.zeros([height, width, 1], dtype=np.uint8)
        cv.drawContours(seg_mask, [seg], -1, (255), thickness=cv.FILLED)
        while True:
            center = (random.randint(0, bg_width),
                      random.randint(0, bg_height))
            if cv.pointPolygonTest(roi, center, False) >= 0:
                break
        pix_cnt = 0
        roi_img = np.zeros(bg.shape, dtype=np.uint8)
        roi_mask = np.zeros([bg_height, bg_width, 1], dtype=np.uint8)
        cv.fillPoly(roi_mask, [roi], (255), 1)
        cv.copyTo(bg, roi_mask, roi_img)
        for h in range(bbox[3]):
            for w in range(bbox[2]):
                if bbox[1]+h >= height or bbox[0]+w >= width:
                    continue
                if cv.pointPolygonTest(seg, [int(bbox[0]+w), int(bbox[1]+h)], False) < 0:
                    continue
                if cv.pointPolygonTest(roi, [center[0]+w, center[1]+h], False) >= 0:
                    roi_img[center[1]+h, center[0] +
                            w] = img[bbox[1]+h, bbox[0]+w]
                    pix_cnt = pix_cnt+1
        if (roi_type == "large" and pix_cnt > area/10.) or (roi_type == "small" and pix_cnt > area/5.):
            return roi_img
        else:
            return None

    def getitem(self, cls):
        # 随机裁切背景
        bg_idx = np.random.randint(0, len(self.bg_path))
        bg_name = self.bg_path[bg_idx]
        # bg_idx = 38
        self.bg = cv.imread(os.path.join(self.bg_root, bg_name))
        if self.bg is None:
            return (np.array([0]), np.array([0])), -1

        height, width, _ = self.bg.shape
        cam_name = self.bg_path[bg_idx].split("-")[0]
        rois = self.cam_rois[cam_name]
        roi_idx = np.random.randint(0, len(rois))
        # roi_idx = 1

        roi = rois[roi_idx].copy()
        bg_img = self.bg.copy()
        roi_bbox = cv.boundingRect(roi)
        vx, vy, x, y = cv.fitLine(roi, cv.DIST_L2, 0, 0.01, 0.01)
        # 沿向
        radian = np.arctan2(vy, vx)
        radian2 = np.arctan2(vx, vy)
        # resize
        if roi_bbox[2]*roi_bbox[3] > 1e4:
            roi_type = "large"
            offset_dist = np.random.randint(-50, 50)
            roi_offset_x = offset_dist * math.cos(radian)
            roi_offset_y = offset_dist * math.sin(radian)
            offset_dist = np.random.randint(-10, 10)
            roi_offset_x = roi_offset_x+offset_dist * math.cos(radian2)
            roi_offset_y = roi_offset_y+offset_dist * math.sin(radian2)
        else:
            roi_type = "small"
            offset_dist = np.random.randint(-7, 7)
            roi_offset_x = offset_dist * math.cos(radian)
            roi_offset_y = offset_dist * math.sin(radian)
            offset_dist = np.random.randint(-3, 3)
            roi_offset_x = roi_offset_x+offset_dist * math.cos(radian2)
            roi_offset_y = roi_offset_y+offset_dist * math.sin(radian2)

        # print("拟合线段弧度：", radian, roi_offset_x, roi_offset_y)
        roi[:, 0] = np.minimum(np.maximum(roi[:, 0] + roi_offset_x, 0), width)
        roi[:, 1] = np.minimum(np.maximum(roi[:, 1] + roi_offset_y, 0), height)
        roi_bbox = cv.boundingRect(roi)
        roi_img = np.zeros(bg_img.shape, dtype=np.uint8)
        roi_mask = np.zeros([height, width, 1], dtype=np.uint8)
        cv.fillPoly(roi_mask, [roi], (255), 1)
        cv.copyTo(bg_img, roi_mask, roi_img)
        roi_img = roi_img[roi_bbox[1]:roi_bbox[1] + roi_bbox[3],
                          roi_bbox[0]:roi_bbox[0] + roi_bbox[2], ...]
        # roi_mask = roi_mask[roi_bbox[1]:roi_bbox[1]+roi_bbox[3], roi_bbox[0]:roi_bbox[0]+roi_bbox[2], ...]
        roi[:, 0] = np.maximum(roi[:, 0] - roi_bbox[0], 0)
        roi[:, 1] = np.maximum(roi[:, 1] - roi_bbox[1], 0)

        # print("idx:", bg_idx, roi_idx, method)
        tmp_img = roi_img.copy()
        if cls == 0:
            draw_cnt = np.random.randint(0, 4)
            while draw_cnt:
                draw_cnt -= 1
                tmp_img = self.fake_shadow(tmp_img, roi)
        elif cls == 1:
            cls = 0
            draw_cnt = np.random.randint(1, 4)
            while draw_cnt:
                method = np.random.randint(0, 2)
                if method == 0:
                    draw_cnt -= 1
                    while True:
                        anno_idx = np.random.randint(0, self.anno_len)
                        processed_img = self.mosic(tmp_img, anno_idx, roi)
                        if processed_img is not None:
                            tmp_img = processed_img
                            cls = 1
                            break
                elif method == 1:
                    draw_cnt -= 1
                    tmp_img = self.fake_shadow(tmp_img, roi)
        if 0:
            roi_texture_image = cv.resize(tmp_img, (224, 224))
        else:
            # 计算填充后的图像大小
            target_size = 224
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
            # 计算填充区域的位置
            pad_top = (target_size - new_h) // 2
            # pad_bottom = target_size - new_h - pad_top
            pad_left = (target_size - new_w) // 2
            # pad_right = target_size - new_w - pad_left
            # 将图像嵌入到填充后的图像中心
            roi_texture_image[pad_top:pad_top + new_h,
                              pad_left:pad_left + new_w] = resized_image

        # 将生成的roi伪数据填充回原图
        bg_texture_image = bg_img.copy()
        texture_image = np.zeros(bg_img.shape, dtype=np.uint8)
        texture_image[roi_bbox[1]:roi_bbox[1]+roi_bbox[3],
                      roi_bbox[0]:roi_bbox[0]+roi_bbox[2], ...] = tmp_img
        cv.copyTo(texture_image, roi_mask, bg_texture_image)
        return roi_img, roi_texture_image, bg_texture_image, cls, bg_name

    def __len__(self):
        return len(self.bg_path)


cam_rois = {"cam113": [[1333, 243, 555, 105]],
            "cam124": [[479, 245, 598, 131], [1062, 200, 109, 26]],
            "cam157": [[557, 243, 597, 116], [1120, 207, 109, 26]],
            "cam168": [[1309, 280, 528, 98], [779, 249, 105, 22]],
            "cam213": [[1460, 200, 515, 112], [913, 153, 105, 22]],
            "cam224": [[723, 261, 565, 109], [1321, 232, 107, 30]],
            "cam257": [[740, 232, 547, 94], [1308, 206, 109, 26]],
            "cam268": [[1327, 213, 388, 91], [809, 182, 102, 21]],
            "cam313": [[1402, 217, 520, 95], [849, 184, 99, 17]],
            "cam324": [[626, 251, 571, 96], [1220, 220, 103, 19]],
            "cam357": [[771, 288, 535, 77], [1332, 275, 95, 16]],
            "cam368": [[1298, 255, 485, 82], [775, 230, 92, 17]],
            "cam413": [[1428, 262, 474, 88], [911, 227, 93, 17]],
            "cam424": [[226, 309, 452, 136], [748, 221, 96, 17]],
            "cam457": [[516, 282, 552, 114], [1118, 231, 96, 21]],
            "cam468": [[921, 117, 526, 68], [372, 159, 96, 29]],
            "cam513": [[1181, 151, 478, 73], [665, 141, 88, 15]],
            "cam524": [[563, 283, 508, 95], [1121, 244, 93, 15]],
            "cam557": [[415, 340, 539, 113], [1003, 286, 102, 19]],
            "cam568": [[1417, 286, 515, 92], [861, 253, 102, 15]],
            "cam624": [[766, 218, 527, 73]],
            "cam657": [[432, 298, 542, 113], [1029, 246, 102, 19]],
            "cam668": [[1503, 211, 503, 102], [948, 165, 102, 19]]}

root = "/home/crrcdt123/datasets2/二门防夹数据/s8-20240307_src/"
forge_root = "/home/crrcdt123/datasets/yolov8/datasets/coco/"
save_root = "/home/crrcdt123/datasets2/二门防夹数据/s8-20240307/resize_nopad"
shadow_ = GenerateTextureData(root, forge_root, 1)

abnormal_cnt = 0
good_cnt = 0
uplimit = 100
while True:
    cls = np.random.randint(0, 2)
    print("fake_cnt: %d, good_cnt: %d" % (abnormal_cnt, good_cnt))
    if abnormal_cnt == uplimit and good_cnt == uplimit:
        break
    if abnormal_cnt >= uplimit and cls == 1:
        cls = 0
    elif good_cnt >= uplimit and cls == 0:
        cls = 1
    roi_img, roi_texture_image, texture_image, result_cls, bg_name = shadow_.getitem(
        cls)
    print(cls, result_cls)
    cv.imshow("roi", roi_img)
    cv.imshow("roi_texture", roi_texture_image)
    cv.imshow("texture", texture_image)
    cv.waitKey(0)
    if cls == 0 and good_cnt < uplimit:
        cv.imwrite(os.path.join(save_root, "good",
                                str(good_cnt) + ".jpg"), roi_texture_image)
        good_cnt = good_cnt + 1
    elif cls == 1 and abnormal_cnt < uplimit:
        cv.imwrite(os.path.join(save_root, "abnormal",
                                str(abnormal_cnt) + ".jpg"), roi_texture_image)
        abnormal_cnt = abnormal_cnt + 1
    cv.imwrite(os.path.join(save_root, "raw",
                            bg_name), texture_image)
