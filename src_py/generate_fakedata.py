import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import os 
import random
import cv2 as cv
import torch
from torchvision import transforms
VOC_CLASSES = (
    "person",
    "box",
    "ironshoes",
    "suitcase",
    "safetyhelmet",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
)

class SiameseShadow(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, root, mode, transform):
        super().__init__()
        self.root = root
        self.train = mode
        self.transform = transform
        self.shadow = 0
        self.fake = 0
        if self.train:
            self._annodir = os.path.join(root, "Annotations_not_care")
            self._imgdir = os.path.join(root, "JPEGImages_not_care")
            self._annopath = os.path.join(self._annodir, "%s.xml")
            self._imgpath = os.path.join(self._imgdir, "%s.jpg")
            self.ids = [name.replace(".xml", "") for name in os.listdir(self._annodir)]
            ori_length = len(self.ids)
            for idx, id in enumerate(self.ids):
                if os.path.exists(os.path.join(self._annopath % id)) and \
                  os.path.exists(os.path.join(self._imgpath % id)):
                    tree = ET.parse(self._annopath % id)
                    objs = tree.findall('object')
                    if len(objs) == 0:
                        print("del------")
                        del self.ids[idx]
                else:
                    print("del------")
                    del self.ids[idx]
            del self.ids[0:1000]
            now_length = len(self.ids)
            print("length: %d --> %d" % (ori_length, now_length))
            self.bg_name = "good"
        else:
            # self.mosic_path = os.listdir(os.path.join(self.root, "val_mosic"))
            # self.shadow_path = os.listdir(os.path.join(self.root, "val_shadow"))
            self.bg_name = "abnormal"
        self.bg_root = "/home/crrcdt123/datasets2/二门防夹数据/车间背景/cam157/"
        self.bg_path = os.listdir(os.path.join(self.bg_root, self.bg_name))
        # self.bg = cv.imread("/home/crrcdt123/git/siamese-triplet/data/background.jpg")
        # self.bg = cv.resize(self.bg, (256, 256))

    def load_anno_from_ids(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()

        assert self.target_transform is not None
        res, img_info = self.target_transform(target)
        height, width = img_info

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        resized_info = (int(height * r), int(width * r))

        return (res, img_info, resized_info)
    
    def fake_shadow(self, img):
        size = img.shape
        shadow_img = img.copy()
        for i in range(2):
            mask = img.copy()
            method = random.randint(0, 3)
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
                startAng, endAng = random.randint(0, 360), random.randint(0, 360)
                cv.ellipse(mask, (int(center[0]), int(center[1])),
                           (int(axes[0]), int(axes[1])), angle, startAng, endAng, (0, 0, 0), -1)
                shadow_img = cv.addWeighted(shadow_img, alpha, mask, blta, 0)
        return shadow_img

    def mosic(self, img1, index):
        img_id = self.ids[index]
        tree = ET.parse(self._annopath % img_id)
        objs = tree.findall('object')
        idx = np.random.randint(0, len(objs))
        obj = objs[idx]
        name = obj.find('name')
        if str(name.text) not in VOC_CLASSES:
            return None
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))

        # load img  shape[row=height, col=width]
        img2 = cv.imread(self._imgpath % img_id)
        roi = img2[ymin:ymax, xmin:xmax, :]
        # resize rotate
        scale = np.random.randint(30, 100) / 100.
        roi_shape = np.array(roi.shape[:2])*scale
        roi_shape = roi_shape.astype(np.int32)
        if roi_shape[0] % 2 != 0:
            roi_shape[0] = roi_shape[0] + 1
        if roi_shape[1] % 2 != 0:
            roi_shape[1] = roi_shape[1] + 1
        cv.resize(roi, (roi_shape[0], roi_shape[1]), roi)
        center = [np.random.randint(0, img1.shape[0]),
                  np.random.randint(0, img1.shape[1])]  # [y, x]
        # 上 下 左 右
        valid_roi_shape = [abs(min(0, center[0] - roi_shape[0]//2)),
                           roi_shape[0] if (center[0] + roi_shape[0]//2) < img1.shape[0]
                           else roi_shape[0]//2 + img1.shape[0] - center[0],
                           abs(min(0, center[1] - roi_shape[1]//2)),
                           roi_shape[1] if (center[1] + roi_shape[1]//2) < img1.shape[1]
                           else roi_shape[1]//2 + img1.shape[1] - center[1]
                          ]
        inImg_roi_shape = [max(0, center[0] - roi_shape[0]//2),
                           min(img1.shape[0], center[0] + roi_shape[0]//2),
                           max(0, center[1] - roi_shape[1]//2),
                           min(img1.shape[1], center[1] + roi_shape[1]//2)]
        if ((valid_roi_shape[1] - valid_roi_shape[0]) *
            (valid_roi_shape[3] - valid_roi_shape[2])) > 30:
            img1[inImg_roi_shape[0]: inImg_roi_shape[1],
                 inImg_roi_shape[2]: inImg_roi_shape[3],
                 :] = roi[valid_roi_shape[0]: valid_roi_shape[1],
                          valid_roi_shape[2]: valid_roi_shape[3], :]
            return img1
        else:
            return None
            
    def __getitem__(self, index):
        # 随机裁切背景
        num_bg = np.random.randint(0, len(self.bg_path))
        self.bg = cv.imread(os.path.join(self.bg_root, self.bg_name, self.bg_path[num_bg]))
        height, width, _ = self.bg.shape
        self.bg = cv.resize(self.bg, (width, 300))
        height, width, _ = self.bg.shape
        roi = [np.random.randint(0, height - 250), np.random.randint(0, width - 250), 244, 244]
        self.bg = self.bg[roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3], ...]

        if self.train:
            target = np.random.randint(0, 3)
            img1 = self.bg.copy()
            if target == 1:
                img2 = self.fake_shadow(img1)
            elif target == 0 or target == 2:
                img2 = self.mosic(img1, index)
                if img2 is None:
                    target = -1
                    img2 = img1
                elif target == 2:
                    img2 = self.fake_shadow(img2)
                    target = 0
        else:
            img1 = self.bg.copy()
            img2 = img1.copy()
            target = 1
            # if index < 100:
            #     img2 = cv.imread(os.path.join(self.root, "val_good", self.shadow_path[index]))
            #     target = 1
            # else:
            #     img2 = cv.imread(os.path.join(self.root, "val_abnormal", self.mosic_path[index - 100]))
            #     target = 0
            
            # img2 = cv.resize(img2, (224, 224))
        # if self.fake == 100 and self.shadow == 100:
        #     return
        # cv.imshow("img2", img2)
        # cv.waitKey(1)
        # img1 = Image.fromarray(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
        # img2 = Image.fromarray(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
        # if self.transform is not None:
        #     img1 = self.transform(img1)
        #     img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        if self.train:
            return len(self.ids)
        else:
            return 100


root = "/home/crrcdt123/datasets/YOLOX/data/VOCdevkit/VOC2007/"
save_root = "/home/crrcdt123/datasets2/二门防夹数据/车间背景/test/"
mean, std = 0.1307, 0.3081
shadow_ = SiameseShadow(root, 0, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                            ]))
train_loader = torch.utils.data.DataLoader(shadow_, batch_size=1, shuffle=True)

abnormal_cnt = 0
good_cnt = 0
uplimit = 100
while True:
    for batch_idx, (data, target) in enumerate(train_loader):
        if target == -1:
            continue
        img = data[1].squeeze().numpy()
        # img = img.queue()
        if target == 0 and abnormal_cnt < uplimit:
            cv.imwrite(os.path.join(save_root, "abnormal",
                                    str(abnormal_cnt) + ".jpg"), img)
            abnormal_cnt = abnormal_cnt + 1
        elif target == 1 and good_cnt < uplimit:
            cv.imwrite(os.path.join(save_root, "good",
                                    str(good_cnt) + ".jpg"), img)
            good_cnt = good_cnt + 1
        else:
            print("已超出上限: %d" % target)
            if abnormal_cnt == uplimit and good_cnt == uplimit:
                break
        print("fake_cnt: %d, good_cnt: %d" % (abnormal_cnt, good_cnt))
    if abnormal_cnt == uplimit and good_cnt == uplimit:
        break
      