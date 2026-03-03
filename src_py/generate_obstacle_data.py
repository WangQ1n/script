import os
import json
import random
import numpy as np
import cv2
from pycocotools.coco import COCO
from skimage.exposure import match_histograms
from scipy.ndimage import gaussian_filter


class MetroSyntheticGenerator:
    def __init__(self, config_path):
        """
        初始化地铁合成数据生成器
        :param config_path: 配置文件路径
        """
        with open(config_path) as f:
            self.config = json.load(f)

        # 加载COCO数据集
        self.coco = COCO(self.config['coco_ann_path'])
        self._init_category_mapping()

        # 加载地铁背景库
        self.bg_imgs = self._load_backgrounds()

        # 预计算轨道掩模
        self.rail_mask = self._generate_rail_template()

        # 初始化增强参数
        self.light_params = {
            'ambient': 0.4,
            'spotlights': [
                {'position': (0.3, 0.1), 'intensity': 0.8, 'falloff': 0.015},
                {'position': (0.7, 0.1), 'intensity': 0.6, 'falloff': 0.02}
            ]
        }

    def _init_category_mapping(self):
        """初始化COCO到地铁的类别映射"""
        self.cat_map = {}
        for coco_cat in self.coco.loadCats(self.coco.getCatIds()):
            self.cat_map[coco_cat['id']] = self.config['category_mapping'].get(
                coco_cat['name'], 'unknown'
            )

    def _load_backgrounds(self):
        """加载地铁背景图像库"""
        bg_imgs = []
        for bg_file in os.listdir(self.config['background_dir']):
            img = cv2.imread(os.path.join(
                self.config['background_dir'], bg_file))
            if img is not None:
                bg_imgs.append(img)
        return bg_imgs

    def _generate_rail_template(self, img_size=(1080, 1920)):
        """生成轨道区域模板"""
        mask = np.zeros(img_size, dtype=np.uint8)
        cv2.rectangle(mask,
                      (int(img_size[1]*0.1), int(img_size[0]*0.7)),
                      (int(img_size[1]*0.9), int(img_size[0]*0.95)),
                      255, -1)
        return mask

    def _get_random_foreground(self):
        """随机获取COCO前景目标"""
        valid_cats = [
            k for k, v in self.config['category_mapping'].items() if v != 'ignore']
        cat_ids = self. coco.getCatIds(catNms=valid_cats)
        img_ids = self.coco.getImgIds(catIds=cat_ids)

        while True:
            img_id = random.choice(img_ids)
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
            anns = self.coco.loadAnns(ann_ids)

            if len(anns) > 0:
                ann = random.choice(anns)
                if len(ann['segmentation']) > 1:
                    continue
                if ann['area'] > self.config['min_obj_area']:
                    break

        # 加载图像和mask
        img_data = self.coco.loadImgs(ann['image_id'])[0]
        print("image_id", ann['image_id'])
        img_path = os.path.join(
            self.config['coco_img_dir'], img_data['file_name'])
        img = cv2.imread(img_path)
        mask = self.coco.annToMask(ann)
        cv2.imshow("img_data", img)
        return img, mask, ann

    def _adapt_perspective(self, img, mask, bg_size):
        """适配地铁轨道视角的透视变换"""
        h, w = img.shape[:2]

        # 生成随机透视变换
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = np.float32([
            [random.uniform(0, w*0.3), random.uniform(0, h*0.2)],
            [random.uniform(w*0.7, w), random.uniform(0, h*0.2)],
            [random.uniform(w*0.6, w), random.uniform(h*0.8, h)],
            [random.uniform(0, w*0.4), random.uniform(h*0.8, h)]
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # 应用变换
        warped_img = cv2.warpPerspective(
            img, M, bg_size[::-1],
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        warped_mask = cv2.warpPerspective(
            mask, M, bg_size[::-1],
            flags=cv2.INTER_NEAREST
        )

        return warped_img, warped_mask

    def _resize_with_padding(self, img, target_w, target_h):
        """保持长宽比的填充调整"""
        h, w = img.shape[:2]
        ratio = min(target_w / w, target_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 计算填充
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        # 边界填充（使用镜像反射）
        padded = cv2.copyMakeBorder(
            resized, 
            pad_h, target_h - new_h - pad_h,
            pad_w, target_w - new_w - pad_w,
            cv2.BORDER_REFLECT_101
        )
        return padded

    def _apply_metro_lighting(self, img, mask, bg_img):
        """应用地铁环境光照特性"""
        # 1. 调整参考图像尺寸（匹配输入图像大小）
        if bg_img.shape[:2] != img.shape[:2]:
            # 方法1：智能填充（保持内容比例）
            bg_img = self._resize_with_padding(bg_img, img.shape[1], img.shape[0])
        # 1. 颜色迁移
        matched = match_histograms(img, bg_img, channel_axis=True)

        # 2. 光照衰减模拟
        light_map = np.zeros(
            img.shape[:2], dtype=np.float32) + self.light_params['ambient']
        for light in self.light_params['spotlights']:
            x = int(light['position'][0] * img.shape[1])
            y = int(light['position'][1] * img.shape[0])
            dist = np.sqrt(
                (np.arange(img.shape[1]) - x)**2 +
                (np.arange(img.shape[0])[:, None] - y)**2
            )
            light_map += light['intensity'] * np.exp(-light['falloff'] * dist)

        # 3. 应用光照
        img_float = matched.astype(np.float32)
        for c in range(3):
            img_float[:, :, c][mask > 0] *= light_map[mask > 0]
        img_adapted = np.clip(img_float, 0, 255).astype(np.uint8)

        return img_adapted

    def _add_physical_effects(self, img, mask):
        """添加物理效果（修正数据类型问题）"""
        # 输入验证
        assert img.dtype == mask.dtype, f"类型不匹配: 图像{img.dtype} vs 掩码{mask.dtype}"
        
        # 1. 转换统一类型（推荐float32计算）
        img_float = img.astype(np.float32) / 255.0
        mask_float = mask.astype(np.float32) / 255.0
        
        # 2. 接触阴影计算
        contours, _ = cv2.findContours(
            (mask_float * 255).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        shadow = np.zeros_like(mask_float)
        for cnt in contours:
            bottom = tuple(cnt[cnt[:,:,1].argmax()][0])
            if self.rail_mask[bottom[1], bottom[0]] > 0:
                cv2.circle(shadow, bottom, 20, 0.4, -1)
        
        # 3. 运动模糊（保持float32计算）
        if random.random() > 0.5:
            ksize = random.randint(5, 15)
            kernel = np.zeros((ksize, ksize), dtype=np.float32)
            direction = random.choice([0, 45, 90])
            if direction == 0:
                kernel[ksize//2, :] = 1.0/ksize
            elif direction == 45:
                kernel = np.eye(ksize, dtype=np.float32) / ksize
            else:
                kernel[:, ksize//2] = 1.0/ksize
            
            blurred = cv2.filter2D(img_float, -1, kernel)
            img_float[mask_float > 0.5] = blurred[mask_float > 0.5]
        
        # 4. 粉尘效果（使用float32计算）
        if random.random() > 0.7:
            dust = (np.random.normal(0, 0.05, img.shape[:2]) > 0.03).astype(np.float32) * 0.3
            img_float = np.clip(img_float + dust[..., None], 0, 1)
        
        # 5. 合成阴影（显式指定输出类型）
        shadow_rgb = np.repeat(shadow[..., None], 3, axis=2)
        result = cv2.addWeighted(
            img_float, 1.0,
            shadow_rgb, 1.0,
            0.0,  # gamma
            dtype=cv2.CV_32F  # 显式指定输出类型
        )
        
        # 转换回uint8
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    def generate_sample(self):
        """生成一个合成样本"""
        # 1. 随机选择背景
        bg = random.choice(self.bg_imgs)
        bg_h, bg_w = bg.shape[:2]
        if bg_h != 1080:
            bg = cv2.resize(bg, (1920, 1080))
            bg_h, bg_w = bg.shape[:2]
        # 2. 获取COCO前景
        fg_img, fg_mask, ann = self._get_random_foreground()

        # 3. 调整大小 (保持长宽比)
        while True:
            scale = random.uniform(0.5, 1.2)
            if scale * ann['area'] > 1500:
                break
        fg_h, fg_w = int(fg_img.shape[0] * scale), int(fg_img.shape[1] * scale)
        fg_img = cv2.resize(fg_img, (fg_w, fg_h))
        fg_mask = cv2.resize(fg_mask, (fg_w, fg_h),
                             interpolation=cv2.INTER_NEAREST)

        # 4. 透视变换
        warped_img, warped_mask = self._adapt_perspective(
            fg_img, fg_mask, (bg_h, bg_w))

        # 5. 光照迁移
        # bg_roi = cv2.bitwise_and(bg, bg, mask=self.rail_mask)
        # adapted_img = self._apply_metro_lighting(
        #     warped_img, warped_mask, bg_roi)

        # 6. 物理效果
        # final_fg = self._add_physical_effects(adapted_img, warped_mask)

        # 7. 合成到背景
        result = bg.copy()
        result[warped_mask > 0] = warped_img[warped_mask > 0]
        cv2.imshow("result", result)
        cv2.waitKey(0)
        # 8. 生成标注
        contours, _ = cv2.findContours(
            warped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox = cv2.boundingRect(np.concatenate(
            contours)) if contours else (0, 0, 0, 0)

        return {
            'image': result,
            'mask': warped_mask,
            'bbox': bbox,
            'category': self.cat_map[ann['category_id']],
            'coco_id': ann['id']
        }


# 配置文件示例 (config.json)
"""
{
    "coco_ann_path": "annotations/instances_train2017.json",
    "coco_img_dir": "train2017",
    "background_dir": "metro_backgrounds",
    "min_obj_area": 500,
    "category_mapping": {
        "person": "passenger",
        "backpack": "luggage",
        "suitcase": "luggage",
        "handbag": "luggage",
        "bottle": "hazard",
        "cup": "hazard",
        "tie": "cloth",
        "umbrella": "obstruction"
    }
}
"""

if __name__ == "__main__":
    # 初始化生成器
    generator = MetroSyntheticGenerator("config/config.json")

    # 生成并保存样本
    output_dir = "synthetic_output"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(100):  # 生成100个样本
        sample = generator.generate_sample()

        # 保存图像和标注
        cv2.imwrite(f"{output_dir}/sample_{i}.jpg", sample['image'])
        cv2.imwrite(f"{output_dir}/mask_{i}.png", sample['mask']*255)

        with open(f"{output_dir}/ann_{i}.json", "w") as f:
            json.dump({
                'bbox': sample['bbox'],
                'category': sample['category'],
                'coco_id': sample['coco_id']
            }, f)

        print(f"Generated sample {i} - Category: {sample['category']}")
