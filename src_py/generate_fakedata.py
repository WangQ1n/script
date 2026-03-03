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

        self.load_anno()
        self.load_bg_roi()
        self.load_own_forge()
        files = glob.glob(os.path.join(self.bg_root, '**.jpg'), recursive=False)
        self.bg_path = [os.path.basename(file) for file in files]

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

    def mosic(self, bg, roi, texture, seg, bbox):
        height, width, _ = texture.shape
        roi_bbox = cv.boundingRect(roi)
        area = cv.contourArea(roi)
        if roi_bbox[2]*roi_bbox[3] > 1e4:
            roi_type = "large"
            scale = np.random.randint(10, 150)/100.
        else:
            roi_type = "small"
            scale = np.random.randint(10, 150)/100.
        seg = seg * scale
        seg = seg.astype(np.int32)
        bbox = np.ceil(bbox * scale)
        bbox = bbox.astype(np.int32)
        texture = cv.resize(texture, (int(width * scale), int(height * scale)))
        height, width, _ = texture.shape
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
            c_h = int(h - bbox[3] / 2)
            for w in range(bbox[2]):
                c_w = int(w - bbox[2] / 2)
                if bbox[1]+h >= height or bbox[0]+w >= width:
                    continue
                if cv.pointPolygonTest(seg, [int(bbox[0]+w), int(bbox[1]+h)], False) < 0:
                    continue
                if cv.pointPolygonTest(roi, [center[0]+c_w, center[1]+c_h], False) >= 0:
                    roi_img[center[1]+c_h, center[0] +
                            c_w] = texture[bbox[1]+h, bbox[0]+w]
                    pix_cnt = pix_cnt+1
        # large:20x20x缩放2.7 = 1080 small:10x10x缩放1.0 = 100
        if (roi_type == "large" and pix_cnt > 1000.) or (roi_type == "small" and pix_cnt > 200.):
            # print("pix_cnt:", pix_cnt)
            return roi_img
        else:
            return None

    # crop
    def crop(self, idx):
        # 随机裁切背景
        bg_idx = idx
        bg_name = self.bg_path[bg_idx]
        self.bg = cv.imread(os.path.join(self.bg_root, bg_name))
        if self.bg is None:
            return (np.array([0]), np.array([0])), -1

        height, width, _ = self.bg.shape
        # cam_name = self.bg_path[bg_idx].split("-2024")[0]
        cam_name = self.bg_path[bg_idx].split("-")[0]
        rois = self.cam_rois[cam_name].copy()
        bg_img = self.bg.copy()
        roi_images = dict()
        for idx, roi in enumerate(rois):
            roi_bbox = cv.boundingRect(roi)
            roi_img = np.zeros(bg_img.shape, dtype=np.uint8)
            roi_mask = np.zeros([height, width, 1], dtype=np.uint8)
            cv.fillPoly(roi_mask, [roi], (255), 1)
            cv.copyTo(bg_img, roi_mask, roi_img)
            roi_img = roi_img[roi_bbox[1]:roi_bbox[1] + roi_bbox[3],
                            roi_bbox[0]:roi_bbox[0] + roi_bbox[2], ...]
            # roi_mask = roi_mask[roi_bbox[1]:roi_bbox[1]+roi_bbox[3], roi_bbox[0]:roi_bbox[0]+roi_bbox[2], ...]
            # roi[:, 0] = np.maximum(roi[:, 0] - roi_bbox[0], 0)
            # roi[:, 1] = np.maximum(roi[:, 1] - roi_bbox[1], 0)

            # print("idx:", bg_idx, roi_idx, method)
            tmp_img = roi_img.copy()
            target_size = 288
            if idx == 0:
                roi_img = cv.resize(tmp_img, (target_size, target_size))
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

            # 将生成的roi伪数据填充回原图
            # bg_texture_image = bg_img.copy()
            # texture_image = np.zeros(bg_img.shape, dtype=np.uint8)
            # texture_image[roi_bbox[1]:roi_bbox[1]+roi_bbox[3],
            #             roi_bbox[0]:roi_bbox[0]+roi_bbox[2], ...] = tmp_img
            # cv.copyTo(texture_image, roi_mask, bg_texture_image)
            roi_images.update({bg_name + "_" + str(idx): roi_img})
        return roi_images
    
    def getitem(self, cls):
        # 随机裁切背景
        roi_img = None
        roi_texture_image = None
        bg_texture_image = None
        bg_idx = np.random.randint(0, len(self.bg_path))
        bg_name = self.bg_path[bg_idx]
        self.bg = cv.imread(os.path.join(self.bg_root, bg_name))
        if self.bg is None:
            return roi_img, roi_texture_image, bg_texture_image, cls, bg_name

        height, width, _ = self.bg.shape
        cam_name = self.bg_path[bg_idx].split("-")[0]
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
            return roi_img, roi_texture_image, bg_texture_image, cls, bg_name
        roi = rois[roi_idx].copy()
        bg_img = self.bg.copy()
        roi_bbox = cv.boundingRect(roi)
        vx, vy, x, y = cv.fitLine(roi, cv.DIST_L2, 0, 0.01, 0.01)

        # 
        src_roi_img = np.zeros(bg_img.shape, dtype=np.uint8)
        roi_mask = np.zeros([height, width, 1], dtype=np.uint8)
        cv.fillPoly(roi_mask, [roi], (255), 1)
        cv.copyTo(bg_img, roi_mask, src_roi_img)
        src_roi_img = src_roi_img[roi_bbox[1]:roi_bbox[1] + roi_bbox[3],
                          roi_bbox[0]:roi_bbox[0] + roi_bbox[2], ...]
        # 沿向
        radian = np.arctan2(vy, vx)
        radian2 = np.arctan2(vx, vy)
        # resize
        if roi_idx == 0:
            # 大区域
            offset_dist = np.random.randint(-25, 25)
            roi_offset_x = offset_dist * math.cos(radian)
            roi_offset_y = offset_dist * math.sin(radian)
            offset_dist = np.random.randint(-5, 5)
            roi_offset_x = roi_offset_x+offset_dist * math.cos(radian2)
            roi_offset_y = roi_offset_y+offset_dist * math.sin(radian2)
        else:
            # 小区域
            offset_dist = np.random.randint(-10, 10)
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

        tmp_img = roi_img.copy()
        if cls == 0:
            draw_cnt = np.random.randint(0, 4)
            while draw_cnt:
                draw_cnt -= 1
                # tmp_img = self.fake_shadow(tmp_img, roi)
        elif cls == 1:
            cls = 0
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
                        # anno_idx = random.choice(list(self.own_anno_info.keys()))
                        # seg, bbox, texture = self.get_own_anno_info(anno_idx)
                        processed_img = self.mosic(tmp_img, roi, texture, seg, bbox)
                        if processed_img is not None:
                            tmp_img = processed_img
                            cls = 1
                            break
                elif method == 1:
                    draw_cnt -= 1
                    tmp_img = self.fake_shadow(tmp_img, roi)
        
        target_size = 288
        if area_flag == "roi0":
            roi_texture_image = cv.resize(tmp_img, (target_size, target_size))
            kernel_size = np.random.randint(1, 5)
            roi_texture_image = cv.blur(roi_texture_image, (kernel_size, kernel_size))
            src_roi_img = cv.resize(src_roi_img, (target_size, target_size))
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
            kernel_size = np.random.randint(1, 5)
            resized_image = cv.blur(resized_image, (kernel_size, kernel_size))
            # 计算填充区域的位置
            pad_top = (target_size - new_h) // 2
            # pad_bottom = target_size - new_h - pad_top
            pad_left = (target_size - new_w) // 2
            # pad_right = target_size - new_w - pad_left
            # 将图像嵌入到填充后的图像中心
            roi_texture_image[pad_top:pad_top + new_h,
                              pad_left:pad_left + new_w] = resized_image
            
            resized_src_roi_img = cv.resize(src_roi_img, (new_w, new_h))
            src_roi_img = np.zeros(
                (target_size, target_size, 3), dtype=np.uint8)
            src_roi_img[pad_top:pad_top + new_h,
                        pad_left:pad_left + new_w] = resized_src_roi_img
        # 将生成的roi伪数据填充回原图
        bg_texture_image = bg_img.copy()
        texture_image = np.zeros(bg_img.shape, dtype=np.uint8)
        texture_image[roi_bbox[1]:roi_bbox[1]+roi_bbox[3],
                      roi_bbox[0]:roi_bbox[0]+roi_bbox[2], ...] = tmp_img
        cv.copyTo(texture_image, roi_mask, bg_texture_image)
        return src_roi_img, roi_texture_image, bg_texture_image, cls, bg_name

    def getitem2(self):
        # 随机裁切背景
        bg_idx = np.random.randint(0, len(self.bg_path))
        # bg_idx = 38
        bg_name = self.bg_path[bg_idx]
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
            offset_dist = np.random.randint(-50, 50)
            roi_offset_x = offset_dist * math.cos(radian)
            roi_offset_y = offset_dist * math.sin(radian)
            offset_dist = np.random.randint(-10, 10)
            roi_offset_x = roi_offset_x+offset_dist * math.cos(radian2)
            roi_offset_y = roi_offset_y+offset_dist * math.sin(radian2)
        else:
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
        abnormal_img = roi_img.copy()
        method = np.random.randint(0, 2)
        if method == 1:
            is_abnormal = False
            while not is_abnormal:
                method = np.random.randint(0, 2)
                if method == 0:
                    while True:
                        anno_idx = np.random.randint(0, self.anno_len)
                        processed_img = self.mosic(abnormal_img, anno_idx, roi)
                        if processed_img is not None:
                            abnormal_img = processed_img
                            is_abnormal = True
                            break
                elif method == 1:
                    abnormal_img = self.fake_shadow(abnormal_img, roi)
            roi_img = abnormal_img.copy()
        shadow_img = roi_img.copy()
        draw_cnt = np.random.randint(0, 4)
        while draw_cnt:
            draw_cnt -= 1
            shadow_img = self.fake_shadow(shadow_img, roi)

        abnormal_img = roi_img.copy()
        draw_cnt = np.random.randint(1, 4)
        is_abnormal = False
        while draw_cnt > 0 or not is_abnormal:
        # while not is_abnormal:
            method = np.random.randint(0, 2)
            if method == 0:
                draw_cnt -= 1
                while True:
                    anno_idx = np.random.randint(0, self.anno_len)
                    # anno_idx = 777
                    processed_img = self.mosic(abnormal_img, anno_idx, roi)
                    if processed_img is not None:
                        abnormal_img = processed_img
                        is_abnormal = True
                        break
            elif method == 1:
                draw_cnt -= 1
                abnormal_img = self.fake_shadow(abnormal_img, roi)

        roi_shadow_image = cv.resize(shadow_img, (224, 224))
        roi_abnormal_image = cv.resize(abnormal_img, (224, 224))
        roi_img = cv.resize(roi_img, (224, 224))
        img_list = [roi_img, roi_shadow_image, roi_abnormal_image]
        return img_list

    def __len__(self):
        return len(self.bg_path)


# 说明
forge_root = "/home/crrcdt123/datasets/yolov8/datasets/coco/"
root = "/home/crrcdt123/二门数据_dalay_1s/"
label_path = os.path.join(root, "images_labels", "roi_two_long")
bg_path = os.path.join(root, "082810")
# save_path = os.path.join(root, "train_2style")
save_path = os.path.join(root, "val", "good")
# save_root = "/home/crrcdt123/datasets2/twoDoor/s8/obstacle_dataset/test/"
area_flags = ["roi0", "roi1"]

if 0:
    for area_flag in area_flags:
        uplimit = 8000
        good_cnt = 5000
        abnormal_cnt = 5000
        shadow_ = GenerateTextureData(label_path, bg_path, forge_root, 1, area_flag)
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
            if roi_img is None:
                continue
            diff = np.abs(np.int16(roi_texture_image) - np.int16(roi_img)).astype(np.uint8)
            if result_cls == 0 and good_cnt < uplimit:
                cv.imwrite(os.path.join(save_path, "2style_good",
                                        area_flag + "_" + str(good_cnt) + ".jpg"), roi_texture_image)
                good_cnt = good_cnt + 1
            elif result_cls == 1 and abnormal_cnt < uplimit:
                cv.imwrite(os.path.join(save_path, "2style_abnormal",
                                        area_flag + "_" + str(abnormal_cnt) + ".jpg"), roi_texture_image)
                abnormal_cnt = abnormal_cnt + 1
            # print(cls, result_cls)
            # cv.imshow("roi", diff)
            # cv.imshow("roi_texture", roi_texture_image)
            # cv.imshow("texture", texture_image)
            # cv.waitKey(0)
            # cv.imwrite(os.path.join(save_root, "raw",
            #                         bg_name), texture_image)
else:
    # 现场数据抠图保存，供测试用
    shadow_ = GenerateTextureData(label_path, bg_path, forge_root, 1, "")
    data_size = shadow_.__len__()
    cnt = 0
    for i in range(0, data_size):
        roi_images = shadow_.crop(i)
        for name, img in roi_images.items():
            cv.imwrite(os.path.join(save_path,
                                    str(cnt) + ".jpg"), img)
            cnt += 1
