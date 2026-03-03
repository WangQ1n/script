import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from ultralytics import YOLO



def extract_track_mask_yolo(model, image, class_id=0):
    src_h, src_w = image.shape[0:2]
    results = model.predict(image, task='segment')
    masks = results[0].masks.data.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    track_masks = []

    roi_x, roi_y, roi_w, roi_h = 50, 150, src_w-100, src_h-150
    for mask, cls in zip(masks, classes):
        if cls == class_id:
            mask = cv2.resize((mask * 255).astype(np.uint8), (src_w, src_h))
            # 创建全零图像（与输入同尺寸）
            # result = np.zeros_like(mask)
            # 将ROI区域复制到结果图像
            # result[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            track_masks.append(mask)
    return track_masks

def compute_normal_vector(p1, p2):
    """
    计算两点之间的法线方向。返回单位法向量。
    """
    # 计算向量 p1 -> p2
    direction = np.array(p2) - np.array(p1)
    
    # 计算法线：二维向量 [dx, dy] 的法线为 [-dy, dx] 或 [dy, -dx]
    normal = np.array([-direction[1], direction[0]])
    
    # 单位化法线向量
    normal_unit = normal / np.linalg.norm(normal)
    
    return normal_unit

def get_centerline_from_mask(mask, step=20):
    """
    从轨道掩码中每隔 step 行采样一个中心点。
    """
    h, w = mask.shape
    center_points = []
    for y in range(0, h, step):
        row = mask[y]
        x_indices = np.where(row > 0)[0]
        if len(x_indices) > 0:
            center_x = int(np.mean(x_indices))
            center_points.append((center_x, y))
    return center_points

def search_right_point(mask, left_point, normal, img_width, img_height, max_distance=200):
    """
    根据法线方向搜索右侧的轨道点。
    从左侧点出发，沿法线方向搜索指定最大距离范围内的右侧轨道点。
    """
    x_left, y_left = left_point
    for distance in range(1, max_distance + 1):  # 搜索最大距离范围内的点
        # 计算偏移位置
        x_right = int(x_left + normal[0] * distance)
        y_right = int(y_left + normal[1] * distance)
        
        # 如果偏移后的点超出图像边界，返回 None
        if x_right < 0 or x_right >= img_width or y_right < 0 or y_right >= img_height:
            return None
        
        # 检查该位置是否属于轨道区域（掩码中的非零值）
        if mask[y_right, x_right] == 0:
            return (x_right, y_right)
    
    return None  # 如果没有找到合适的右侧点，返回 None

def sample_centerline_from_mask(mask, img_width, img_height, sample_step=50, max_samples=1000):
    """
    从图像掩码中采样轨道中心点，采样左边点，通过法线方向确定右边点，然后计算轨道中心点。
    """
    sampled_left_points = []
    sampled_right_points = []
    right_points = []
    center_points = []
    normal_list = []

    # 从底部向上采样，沿y轴方向逐行提取轨道左侧点
    for y in range(img_height - 1, 0, -sample_step):
        # 找到每行掩码中非零像素的x坐标，并取其平均值作为左侧轨道点
        row = mask[y, :]
        x_indices = np.where(row > 0)[0]
        
        if len(x_indices) > 0:
            left_x = int(np.min(x_indices))  # 取该行掩码非零区域的中点作为左侧点
            left_point = (left_x, y)
            sampled_left_points.append(left_point)
            right_x = int(np.max(x_indices))
            right_point = (right_x, y)
            right_points.append(right_point)
    
    # 根据左边点计算右边点（通过法线方向）
    for i in range(2, len(sampled_left_points)):
        p1, p2, p3 = sampled_left_points[i - 2], sampled_left_points[i - 1], sampled_left_points[i]
        right_p1, right_p2, right_p3 = right_points[i - 2], right_points[i - 1], right_points[i]
        # 计算法线方向
        normal = compute_normal_vector(p1, p3)
        right_normal = compute_normal_vector(right_p1, right_p3)
        # normal = (normal + right_normal) / (np.linalg.norm(normal + right_normal) + 1e-8)
        normal_list.append(normal)
        # 沿法线方向推算右侧点
        right_point = search_right_point(mask, p2, normal, img_width, img_height)
        
        # 检查右侧点是否在图像区域内
        if 0 <= right_point[0] < img_width and 0 <= right_point[1] < img_height:
            sampled_right_points.append(tuple(right_point))
            
            # 计算轨道中心点（左右两点的中点）
            center_x = int((p2[0] + right_point[0]) / 2)
            center_y = int((p2[1] + right_point[1]) / 2)
            center_points.append((center_x, center_y))
        else:
            break  # 如果右侧点超出图像范围，停止采样
    
    return sampled_left_points, sampled_right_points, center_points, normal_list

def extract_centerline_from_mask_by_contour(mask, sample_step=10):
    """
    根据掩码轮廓，从左右边界构造中心线（适用于弯道）
    """
    # 1. 提取轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    # 找最大连通区域（假设为轨道）
    contour = max(contours, key=cv2.contourArea)
    if contour.ndim == 3:
        contour = contour[:, 0, :]  # OpenCV 返回 shape=(N,1,2)
    elif contour.ndim == 2:
        pass  # 已经是 (N, 2)
    else:
        raise ValueError(f"Unexpected contour shape: {contour.shape}")

    # 2. 拆分左右边界
    # 拟合最远点对，作为左右边界初始方向
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    _, (w, h), angle = rect

    # PCA 拟合方向（主方向为轨道延伸方向）
    mean, eigenvecs = cv2.PCACompute(contour.astype(np.float32), mean=None)
    main_direction = eigenvecs[0]  # 主轴

    # 将所有点按主轴方向投影并排序
    proj = contour @ main_direction
    sorted_indices = np.argsort(proj)
    contour_sorted = contour[sorted_indices]

    # 3. 采样并配对中心点
    center_points = []
    for i in range(0, len(contour_sorted) - sample_step, sample_step):
        a = contour_sorted[i]
        b = contour_sorted[i + sample_step]

        # 只保留对之间距离足够大的情况
        if np.linalg.norm(a - b) > 5:
            midpoint = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            center_points.append(midpoint)

    return center_points

def fit_bspline_curve(points, smooth=3):
    if len(points) < 4:
        return np.array(points)

    points = np.array(points)
    x, y = points[:, 0], points[:, 1]

    # 参数tck是一个三元组(t, c, k)
    tck, u = splprep([x, y], s=smooth)
    u_fine = np.linspace(0, 1, num=1000)
    x_fine, y_fine = splev(u_fine, tck)
    return np.vstack((x_fine, y_fine)).T.astype(np.int32)

def fit_bspline_curve_extend_top(points, smooth=3):
    if len(points) < 4:
        return np.array(points)

    points = np.array(points)
    x, y = points[:, 0], points[:, 1]

    # 按 y 坐标升序排列（y 越小越靠近图像顶部）
    idx = np.argsort(y)
    x, y = x[idx], y[idx]

    # 线性外推一个 y=0 的点作为头部
    if y[0] > 0:
        # 用前2~3个点估算斜率，线性回推到 y=0
        dy = y[1] - y[0]
        dx = x[1] - x[0]
        slope = dx / dy if dy != 0 else 0
        new_x = int(x[0] - slope * y[0])
        x = np.insert(x, 0, new_x)
        y = np.insert(y, 0, 0)

    # 拟合B样条曲线
    tck, u = splprep([x, y], s=smooth)
    u_fine = np.linspace(0, 1, num=1000)
    x_fine, y_fine = splev(u_fine, tck)
    return np.vstack((x_fine, y_fine)).T.astype(np.int32)


from scipy.interpolate import splprep, splev
import numpy as np

def fit_bspline_with_extrapolation(points, img_width, img_height, smooth=17, extrapolate_length=0.5):
    """
    使用已知中心点构建B样条，并向前/向后进行外推，直到图像边界。
    extrapolate_length: 外推比例，例如0.2表示在前后各外推20%长度
    """
    points = np.array(points)
    if len(points) < 4:
        return points

    # 判断主方向：决定是在 y 上延伸（纵向轨道）还是 x 上延伸（横向轨道）
    dx = abs(points[-1][0] - points[0][0])
    dy = abs(points[-1][1] - points[0][1])
    is_horizontal = dx > dy

    # 按主方向排序（更平滑）
    points = points[np.argsort(points[:, 0 if is_horizontal else 1])]

    x, y = points[:, 0], points[:, 1]
    
    # 计算均匀参数化
    u = np.linspace(0, 1, len(points))
    # 构建样条曲线
    tck, u = splprep([x, y], k=2)

    # 外推区间 u ∈ [-e, 1+e]
    extend = extrapolate_length
    u_ex = np.linspace(-extend, 1 + extend, 1000)
    x_ex, y_ex = splev(u_ex, tck)

    # 保留在图像内的部分
    valid_mask = (x_ex >= 0) & (x_ex < img_width) & (y_ex >= 0) & (y_ex < img_height)
    curve = np.vstack((x_ex[valid_mask], y_ex[valid_mask])).T

    return curve.astype(np.int32)


def visualize(image, track_masks, spline_points_list, center_points_list, left_points_list, right_points_list, normal_list):
    overlay = image.copy()

    # 绘制每条轨道掩码
    for mask in track_masks:
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(overlay, 1.0, mask_color, 0.4, 0)


    # 绘制拟合（及外推）曲线（红色）
    for spline_points in spline_points_list:
        for pt in spline_points:
            cv2.circle(overlay, tuple(pt), radius=1, color=(255, 0, 255), thickness=2)  # 红色点构成曲线

    # # 绘制采样点（蓝色圆点）
    # for center_points in center_points_list:
    #     for pt in center_points:
    #         cv2.circle(overlay, pt, radius=4, color=(0, 255, 0), thickness=-1)  # 蓝色填充圆点

    # # 绘制采样点（蓝色圆点）
    # for left_points in left_points_list:
    #     for pt in left_points:
    #         cv2.circle(overlay, pt, radius=4, color=(255, 0, 0), thickness=-1)  # 蓝色填充圆点

    # # 绘制采样点（蓝色圆点）
    # for right_points in right_points_list:
    #     for pt in right_points:
    #         cv2.circle(overlay, pt, radius=4, color=(0, 0, 255), thickness=-1)  # 蓝色填充圆点
    for i in range(len(normal_list)):
        for j in range(len(normal_list[i])):
            lp = np.array(left_points_list[i][j+1], dtype=np.int32)
            rp = np.array(right_points_list[i][j], dtype=np.int32)
            cp = center_points_list[i][j]
            normal = np.array(normal_list[i][j])

            # 绘制法线（从左点出发）
            end_pt = (lp + normal * 100).astype(np.int32)
            cv2.arrowedLine(overlay, tuple(lp), tuple(end_pt), (255,255,255), 2, tipLength=0.3)
            # 绘制点
            cv2.circle(overlay, tuple(lp), 4, color=(0, 0, 255), thickness=-1)
            cv2.circle(overlay, tuple(rp), 4, color=(255, 0, 0), thickness=-1)
            cv2.circle(overlay, tuple(cp), 4, color=(0, 255, 0), thickness=-1)

    return overlay
import os
def list_all_images(folder, exts={'.jpg', '.png', '.jpeg', '.bmp', '.tif'}):
    """
    遍历文件夹中所有图片文件，返回文件路径列表
    """
    image_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if os.path.splitext(file)[1].lower() in exts:
                image_paths.append(os.path.join(root, file))
    return image_paths

def main():
    # /home/crrcdt123/git/ultralytics_v8/runs/segment/rail/train_20240329/
    #/home/crrcdt123/git/ultralytics/runs/segment/train6/weights/best.pt
    model_path = '/home/crrcdt123/git/ultralytics_v8/runs/segment/rail/train_20240329/weights/best.pt'

    model = YOLO(model_path)

    folder_path = '/home/crrcdt123/datasets/railway_segmentation/mixed_20240329/images/train/'  # 替换为你的图像目录路径
    image_list = list_all_images(folder_path)
    for img_path in image_list:
        image = cv2.imread(img_path)  # 原图像
        height, width = image.shape[0:2]
        track_masks = extract_track_mask_yolo(model, image, class_id=0)

        if len(track_masks) == 0:
            print("未检测到轨道区域")
            continue
        left_points_list = []
        right_points_list = []
        center_points_list = []
        spline_points_list = []
        normal_list = []
        for mask in track_masks:
            # center_points = get_centerline_from_mask(mask, 50)
            left_points, right_points, center_points, normal = sample_centerline_from_mask(mask, img_width=width, img_height=height, sample_step=20)
            # center_points = extract_centerline_from_mask_by_contour(mask, sample_step=30)
            # spline_points = fit_bspline_curve(center_points, smooth=3)
            # spline_points = fit_bspline_curve_extend_top(center_points, smooth=3)
            spline_points = fit_bspline_with_extrapolation(center_points, img_width=width, img_height=height)
            center_points_list.append(center_points)
            spline_points_list.append(spline_points)
            right_points_list.append(right_points)
            left_points_list.append(left_points)
            normal_list.append(normal)

        result = visualize(image, track_masks, spline_points_list, center_points_list, left_points_list, right_points_list, normal_list)

        cv2.imshow('Multiple Tracks Segmentation and Fitted Centerlines', result)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
