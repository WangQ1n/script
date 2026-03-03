import re
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 这些参数可以根据原C++类中定义调整
prev_hsv_img = None
curr_hsv_img = None
kernel_size = {0: 3, 1: 5, 2: 7}
mode = 0

color_palette = [
    (128, 77, 207), (65, 32, 208), (0, 224, 45), (3, 141, 219),
    (80, 239, 253), (239, 184, 12), (7, 144, 145), (161, 88, 57),
    (0, 166, 46), (218, 113, 53), (193, 33, 128), (190, 94, 113),
    (113, 123, 232), (69, 205, 80), (18, 170, 49), (89, 51, 241),
    (153, 191, 154), (27, 26, 69), (20, 186, 194), (210, 202, 167),
    (196, 113, 204), (9, 81, 88), (191, 162, 67), (227, 73, 120),
    (177, 31, 19), (133, 102, 137), (146, 72, 97), (145, 243, 208),
    (2, 184, 176), (219, 220, 93), (238, 253, 234), (197, 169, 160),
    (204, 201, 106), (13, 24, 129), (40, 38, 4), (5, 41, 34),
    (46, 94, 129), (102, 65, 107), (27, 11, 208), (191, 240, 183),
    (225, 76, 38), (193, 89, 124), (30, 14, 175), (144, 96, 90),
    (181, 186, 86), (102, 136, 34), (158, 71, 15), (183, 81, 247),
    (73, 69, 89), (123, 73, 232), (4, 175, 57), (87, 108, 23),
    (105, 204, 142), (63, 115, 53), (105, 153, 126), (247, 224, 137),
    (136, 21, 188), (122, 129, 78), (145, 80, 81), (51, 167, 149),
    (162, 173, 20), (252, 202, 17), (10, 40, 3), (150, 90, 254),
    (169, 21, 68), (157, 148, 180), (131, 254, 90), (7, 221, 102),
    (19, 191, 184), (98, 126, 199), (210, 61, 56), (252, 86, 59),
    (102, 195, 55), (160, 26, 91), (60, 94, 66), (204, 169, 193),
    (126, 4, 181), (229, 209, 196), (195, 170, 186), (155, 207, 148)
]


def HSVDiff(prev_hsv, curr_hsv, mask, radius):
    """
    bbox: (x, y, w, h)
    mask: uint8 二值图像, 非零为有效区域
    """
    radius = (radius - 1) // 2
    mask_nonzero = cv2.countNonZero(mask)
    if mask_nonzero == 0:
        return 0.0

    h_diff_list = []
    s_diff_list = []
    v_diff_list = []
    for j in range(prev_hsv.shape[0]):
        for i in range(prev_hsv.shape[1]):
            if mask[j, i] == 0:
                continue

            startX = max(0, i - radius)
            endX = min(prev_hsv.shape[1] - 1, i + radius)
            startY = max(0, j - radius)
            endY = min(prev_hsv.shape[0] - 1, j + radius)

            prev_color = prev_hsv[j, i]
            min_dist_l2 = 9999
            min_color = []
            for yy in range(startY, endY + 1):
                for xx in range(startX, endX + 1):
                    if mask[yy, xx] == 0:
                        continue
                    curr_color = curr_hsv[yy, xx]
                    h_dist = abs(int(prev_color[0]) - int(curr_color[0]))
                    if h_dist > 90:
                        h_dist = 180 - h_dist
                    dist_l2 = h_dist + abs(int(
                        curr_color[1]) - int(prev_color[1])) + abs(int(curr_color[2]) - int(prev_color[2]))
                    if dist_l2 < min_dist_l2:
                        min_dist_l2 = dist_l2
                        min_color = curr_color
            h_dist = abs(int(prev_color[0]) - int(min_color[0]))
            if h_dist > 90:
                h_dist = 180 - h_dist
            h_diff_list.append(h_dist)
            s_diff_list.append(int(prev_color[1]) - int(min_color[1]))
            v_diff_list.append(int(prev_color[2]) - int(min_color[2]))
    return h_diff_list, s_diff_list, v_diff_list


def load_images(folder_path, keyword, extensions=(".jpg", ".png", ".jpeg", ".bmp")):
    """
    根据关键字加载指定路径下的图像文件
    :param folder_path: 图像文件夹路径
    :param keyword: 文件名中包含的关键字
    :param extensions: 支持的图像格式
    :return: [(filename, image), ...]
    """
    images_path = []

    if not os.path.exists(folder_path):
        print(f"❌ 路径不存在: {folder_path}")
        return images

    for root, _, files in os.walk(folder_path):
        for file in files:
            if keyword in file and file.lower().endswith(extensions):
                img_path = os.path.join(root, file)
                images_path.append(img_path)
                # img = cv2.imread(img_path)
                # if img is not None:
                #     images.append((file, img))
                # else:
                #     print(f"⚠️ 无法读取图像: {img_path}")

    print(f"✅ 共找到 {len(images_path)} 张匹配关键字 '{keyword}' 的图像。")
    return images_path


def match_frame_by_time(src_str, paths, time_dist):
    """
    匹配前一帧（不跨天）
    :param src_str: 当前文件名字符串，如 "cam113-End-20250926-192259498"
    :param paths: 候选文件路径列表
    :return: (是否匹配成功, 匹配到的路径)
    """

    match_path = ""

    # ---- 提取源文件信息 ----
    ori_data = 0
    ori_time = 0
    ori_name = ""
    found = src_str.rfind("-")
    if found != -1:
        try:
            ori_name = src_str[found - 19:found - 13]   # 名称部分 (6位)
            ori_data = int(src_str[found - 8:found])    # 日期部分 YYYYMMDD
            ori_time = int(src_str[found + 1:found + 7])  # 时间部分 HHMMSS
        except ValueError:
            return False, ""

    min_dist = -999999999
    min_idx = -1

    # ---- 遍历所有候选路径 ----
    for i, match in enumerate(paths):
        found = match.rfind("-")
        if found == -1:
            continue

        try:
            match_name = match[found - 21:found - 15]
            match_data = int(match[found - 8:found])
        except ValueError:
            continue

        if match_data != ori_data or match_name != ori_name:
            continue

        try:
            match_time = int(match[found + 1:found + 7])
        except ValueError:
            continue

        dist = match_time - ori_time
        if dist < 0 and dist > min_dist:
            min_dist = dist
            min_idx = i

    # ---- 判断匹配结果 ----
    if min_idx >= 0 and min_dist > time_dist:
        print(f"time dist:{min_dist}, path:{paths[min_idx]}")
        match_path = paths[min_idx]
        return True, match_path
    else:
        return False, ""


def crop_masked_region(img, img2, mask):
    """
    根据掩码裁切原图的最小外接矩形区域。

    参数:
        img: 原图 (H, W, C)
        mask: 掩码 (H, W)，非零为目标区域

    返回:
        roi_img: 裁切后的图像区域
        roi_mask: 对应裁切的掩码区域
        bbox: 外接矩形 (x_min, y_min, x_max, y_max)
    """
    # 找到掩码非零像素坐标
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None, None, None  # 没有掩码区域

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # 裁切原图和掩码
    roi_img = img[y_min:y_max+1, x_min:x_max+1]
    roi_img2 = img2[y_min:y_max+1, x_min:x_max+1]
    roi_mask = mask[y_min:y_max+1, x_min:x_max+1]

    bbox = (x_min, y_min, x_max, y_max)
    return roi_img, roi_img2, roi_mask, bbox


class Object:
    def __init__(self, id, confidence, bbox, area, contour,
                 hit_high_response_score, shadow_score, contour_score, roi_type):
        self.id = id
        self.confidence = confidence
        self.bbox = bbox  # (x, y, w, h)
        self.area = area
        self.contour = contour  # [(x, y), ...]
        self.hit_high_response_score = hit_high_response_score
        self.shadow_score = shadow_score
        self.contour_score = contour_score
        self.roi_type = roi_type


def load_objects_from_txt(filename):
    objects = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue

            idx = 0
            id_ = int(parts[idx])
            idx += 1
            confidence = float(parts[idx])
            idx += 1
            x = int(parts[idx])
            y = int(parts[idx+1])
            w = int(parts[idx+2])
            h = int(parts[idx+3])
            bbox = (x, y, w, h)
            idx += 4
            area = float(parts[idx])
            idx += 1
            n_points = int(parts[idx])
            idx += 1

            contour = []
            for _ in range(n_points):
                x_str, y_str = parts[idx].split(',')
                contour.append((int(x_str), int(y_str)))
                idx += 1
            contour = np.array(contour)
            hit_high_response_score = float(parts[idx])
            idx += 1
            shadow_score = float(parts[idx])
            idx += 1
            contour_score = float(parts[idx])
            idx += 1
            if (idx >= len(parts)):
                continue
            roi_type = float(parts[idx])
            idx += 1
            obj = Object(id_, confidence, bbox, area, contour,
                         hit_high_response_score, shadow_score, contour_score, roi_type)
            objects.append(obj)

    print(f"✅ Loaded {len(objects)} objects from {filename}")
    return objects


def analyse(h_diff_list, s_diff_list, v_diff_list):
    '''
    分析项：
    平均值
    方差
    l2
    '''
    h_diff = np.array(h_diff_list)
    s_diff = np.array(s_diff_list)
    v_diff = np.array(v_diff_list)
    h_mean_abs = np.mean(np.abs(h_diff))
    s_mean_abs = np.mean(np.abs(s_diff))
    v_mean_abs = np.mean(np.abs(v_diff))
    # h_mean = np.mean(h_diff)
    # s_mean = np.mean(s_diff)
    # v_mean = np.mean(v_diff)
    # h_var = np.var(h_diff)
    # s_var = np.var(s_diff)
    # v_var = np.var(v_diff)
    # h_std = np.std(h_diff)
    # s_std = np.std(s_diff)
    # v_std = np.std(v_diff)
    print("abs平均值:", h_mean_abs, s_mean_abs, v_mean_abs)
    # print("平均值:", h_mean, s_mean, v_mean)
    # print("方差:", h_var, s_var, v_var)
    # print("标准差:", h_std, s_std, v_std)
    return [h_mean_abs, s_mean_abs, v_mean_abs]


def visual(class1, class2):
    class1 = np.array(class1, dtype=np.float32)
    class2 = np.array(class2, dtype=np.float32)
    # 拆分出 H, S, V 三个维度
    # H = data[:, 0]
    # S = data[:, 1]
    # V = data[:, 2]
    # class1 = np.array([
    #     [10, 120, 200],
    #     [20, 150, 220],
    #     [40, 180, 210],
    #     [60, 200, 230],
    # ], dtype=np.float32)

    # class2 = np.array([
    #     [160, 100, 120],
    #     [180, 130, 150],
    #     [200, 150, 180],
    #     [220, 180, 200],
    # ], dtype=np.float32)

    # 合并数据
    samples = np.vstack((class1, class2))
    labels = np.array([0]*len(class1) + [1]*len(class2))  # 0/1 表示类别

    # === 绘图 ===
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 两类颜色
    colors = np.array(['red', 'blue'])

    # 绘制散点
    for c in np.unique(labels):
        points = samples[labels == c]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   color=colors[c],
                   label=f'Class {c+1}', s=80, alpha=0.8, edgecolor='k')

    # 设置标签
    ax.set_xlabel('Hue (H)')
    ax.set_ylabel('Saturation (S)')
    ax.set_zlabel('Value (V)')
    ax.set_title('HSV 三维分布 (两类样本)')

    ax.legend()
    ax.view_init(elev=25, azim=120)
    plt.tight_layout()
    # plt.show()

def visual_class1(class1):
    class1 = np.array(class1, dtype=np.float32)
    # === 每个样本使用不同颜色 ===
    # 可以直接用 HSV 转换成 RGB，以便颜色与数据直观对应
    # hsv_norm = class1 / [180, 255, 255]  # 归一化到 [0,1]
    # colors = plt.cm.hsv(hsv_norm[:, 0])   # 根据 H 通道取色（也可以混合 HSV 生成彩色）
    colors = np.array([color_palette[i % len(color_palette)] for i in range(len(class1))]) / 255.0
    # === 绘制 3D 散点图 ===
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(class1[:, 0], class1[:, 1], class1[:, 2],
            c=colors, s=60, alpha=0.8, edgecolor='k')

    ax.set_xlabel('Hue (H)')
    ax.set_ylabel('Saturation (S)')
    ax.set_zlabel('Value (V)')
    ax.set_title('HSV 样本三维分布（每点不同颜色）')

    ax.view_init(elev=25, azim=120)
    plt.tight_layout()
    # plt.show()

def draw_objects_info(img, objects, font_scale=0.5):
    font_face = cv2.FONT_HERSHEY_DUPLEX
    thickness = 1
    baseline = 0

    for i, obj in enumerate(objects):
        color = (0, 255, 0)
        color = color_palette[i % len(color_palette)]
        # 绘制轮廓
        contour = np.array(obj.contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(img, [contour], -1, color, 1, lineType=cv2.LINE_8)

        # 准备显示文本
        text = f"{obj.area:.2f}, {obj.confidence:.2f}, {obj.hit_high_response_score:.2f}, " \
               f"{obj.contour_score:.2f}, {obj.shadow_score:.2f}"

        (text_w, text_h), baseline = cv2.getTextSize(
            text, font_face, font_scale, thickness)
        text_org = (10, 100 - i * (text_h + 6))

        # 绘制背景矩形
        cv2.rectangle(img,
                      text_org,
                      (text_org[0] + text_w, text_org[1] -
                       text_h - baseline - 3),
                      color, thickness=-1)

        # 绘制黑色文字
        cv2.putText(img, text,
                    (text_org[0], text_org[1] - baseline),
                    font_face, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)

    return img


def main():
    image_dir = "/media/crrcdt123/glam/crrc/data/su8/2door/anomaly/"
    end_images_path = load_images(image_dir, "End")
    start_images_path = load_images(image_dir, "Start")
    shadow_data = []
    for i, end_image_path in enumerate(end_images_path):
        # if i > 0:
        #     break
        isOk, start_image_path = match_frame_by_time(
            end_image_path, start_images_path, -120)
        if isOk is False:
            continue
        start_image = cv2.imread(start_image_path)
        start_image_hsv = cv2.cvtColor(start_image, cv2.COLOR_BGR2HSV)
        end_image = cv2.imread(end_image_path)
        end_image_hsv = cv2.cvtColor(end_image, cv2.COLOR_BGR2HSV)
        objs = load_objects_from_txt(end_image_path.replace(".jpg", ".txt"))
        height, width = start_image.shape[:2]
        # result_imge = draw_objects_info(end_image, objs)
        # concat_img = np.zeros((result_imge.shape[0], result_imge.shape[1] * 2, result_imge.shape[2]), dtype=result_imge.dtype)
        # concat_img[:, 0:result_imge.shape[1]] = start_image
        # concat_img[:, result_imge.shape[1]:] = result_imge
        # cv2.imshow("concat_img", concat_img)
        # cv2.waitKey(0)
        for obj in objs:
            mask = np.zeros((height, width), dtype=np.uint8)
            mask = cv2.drawContours(mask, [obj.contour], -1, 255, -1)
            roi_start_img, roi_end_img, roi_mask, bbox = crop_masked_region(
                start_image_hsv, end_image_hsv, mask)
            h_diff_list, s_diff_list, v_diff_list = HSVDiff(
                roi_start_img, roi_end_img, roi_mask, 5)
            data = analyse(h_diff_list, s_diff_list, v_diff_list)
            # if obj.roi_type == 1 and obj.hit_high_response_score > 0.1:
            #     obj.hit_high_response_score = 1
            # data[2] = obj.contour_score + obj.hit_high_response_score - obj.shadow_score
            shadow_data.append(data)

    image_dir = "/media/crrcdt123/glam/crrc/data/su8/2door/081701-20250823/pictures_unbluelight/"
    end_images_path = load_images(image_dir, "End")
    start_images_path = load_images(image_dir, "Start")
    anomly_data = []
    for i, end_image_path in enumerate(end_images_path):
        if i > 0:
            break
        isOk, start_image_path = match_frame_by_time(
            end_image_path, start_images_path, -120000)
        if isOk is False:
            continue
        start_image = cv2.imread(start_image_path)
        start_image_hsv = cv2.cvtColor(start_image, cv2.COLOR_BGR2HSV)
        end_image = cv2.imread(end_image_path)
        end_image_hsv = cv2.cvtColor(end_image, cv2.COLOR_BGR2HSV)
        objs = load_objects_from_txt(end_image_path.replace(".jpg", ".txt"))
        height, width = start_image.shape[:2]
        # result_imge = draw_objects_info(end_image, objs)
        # concat_img = np.zeros((result_imge.shape[0], result_imge.shape[1] * 2, result_imge.shape[2]), dtype=result_imge.dtype)
        # concat_img[:, 0:result_imge.shape[1]] = start_image
        # concat_img[:, result_imge.shape[1]:] = result_imge
        # cv2.imshow("concat_img", concat_img)
        cv2.waitKey(0)
        for obj in objs:
            mask = np.zeros((height, width), dtype=np.uint8)
            mask = cv2.drawContours(mask, [obj.contour], -1, 255, -1)
            roi_start_img, roi_end_img, roi_mask, bbox = crop_masked_region(
                start_image_hsv, end_image_hsv, mask)
            h_diff_list, s_diff_list, v_diff_list = HSVDiff(
                roi_start_img, roi_end_img, roi_mask, 5)
            data = analyse(h_diff_list, s_diff_list, v_diff_list)
            # if obj.roi_type == 1 and obj.hit_high_response_score > 0.1:
            #     obj.hit_high_response_score = 1
            # data[2] = obj.contour_score + obj.hit_high_response_score - obj.shadow_score
            anomly_data.append(data)
    visual(shadow_data, anomly_data)
    visual_class1(shadow_data)
    visual_class1(anomly_data)
    plt.show()


if __name__ == '__main__':
    main()
