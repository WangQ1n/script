import cv2
import numpy as np

VIDEO_PATH = "/home/crrcdt123/git/script/keyboard.mp4"
SAVE_PATH = "panorama_result.jpg"
SKIP_FRAMES = 25  # 每隔几帧作为关键帧拼接
FEATURES = 3000

# 初始化
cap = cv2.VideoCapture(VIDEO_PATH)
ret, first_frame = cap.read()
assert ret, "无法读取视频第一帧"

orb = cv2.ORB_create(FEATURES)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 全景初始化
panorama = first_frame.copy()
H_total = np.eye(3)

# 第一帧作为参考帧
ref_frame = first_frame
kp_ref, des_ref = orb.detectAndCompute(ref_frame, None)

# 当前画布尺寸（先设大一点）
canvas_h, canvas_w = 2 * first_frame.shape[0], 4 * first_frame.shape[1]
canvas_center = (canvas_w // 2, canvas_h // 2)

# 画布初始化
panorama_canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 0

offset = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]], dtype=np.float32)

# 将第一帧放到画布中心
warped_first = cv2.warpPerspective(first_frame, offset, (canvas_w, canvas_h))
panorama_canvas = np.minimum(panorama_canvas, warped_first)
def register_feature_based(img1, img2):
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        return H
    return None

def register_optical_flow(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = gray1.shape
    remap_x = np.tile(np.arange(w), (h, 1)).astype(np.float32) + flow[..., 0]
    remap_y = np.tile(np.arange(h)[:, np.newaxis], (1, w)).astype(np.float32) + flow[..., 1]
    aligned = cv2.remap(img1, remap_x, remap_y, interpolation=cv2.INTER_LINEAR)
    return aligned

def register_phase_correlation(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    shift, _ = cv2.phaseCorrelate(gray1, gray2)
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    return cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))

def blend_paste(canvas, warped):
    mask = (warped > 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    canvas[mask == 1] = warped[mask == 1]
    return canvas

def feather_blend(img1, img2, mask1, mask2, corner1, corner2):
    blender = cv2.detail_FeatherBlender()
    blender.setSharpness(1.0)
    blender.prepare([corner1, corner2])
    blender.feed(img1.astype(np.int16), mask1, corner1)
    blender.feed(img2.astype(np.int16), mask2, corner2)
    result, _ = blender.blend(None, None)
    return np.clip(result, 0, 255).astype(np.uint8)

def stitch_with_feather_blend(img1, img2, H):
    # 图像尺寸和掩码
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 变换第二张图像
    corners_img2 = np.float32([[0,0], [w2,0], [w2,h2], [0,h2]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_img2, H)
    [xmin, ymin] = np.int32(warped_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(warped_corners.max(axis=0).ravel() + 0.5)

    # 平移矩阵 + warp
    translation = np.array([[1, 0, -xmin],
                            [0, 1, -ymin],
                            [0, 0, 1]])
    size = (xmax - xmin, ymax - ymin)
    warped_img2 = cv2.warpPerspective(img2, translation @ H, size)

    # warp 第一张图
    warped_img1 = np.zeros_like(warped_img2)
    warped_img1[-ymin:h1 - ymin, -xmin:w1 - xmin] = img1

    # 创建掩码
    mask1 = (cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
    mask2 = (cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255

    # 曝光补偿器
    compensator = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
    compensator.feed([(-xmin, -ymin), (0,0)], [warped_img1, warped_img2], [mask1, mask2])
    compensator.apply(0, (-xmin, -ymin), warped_img1, mask1)
    compensator.apply(1, (0,0), warped_img2, mask2)

    # 羽化融合器
    blender = cv2.detail_FeatherBlender()
    blender.setSharpness(1.0)
    blender.prepare([(-xmin, -ymin), (0,0)])
    blender.feed(warped_img1.astype(np.int16), mask1, (-xmin, -ymin))
    blender.feed(warped_img2.astype(np.int16), mask2, (0,0))

    result, _ = blender.blend(None, None)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def shrink_mask2_rowwise(mask2, shrink_px=2):
    """
    精确对 mask2 的每一行非零段右侧进行 shrink_px 像素的收缩
    例如：行中从右数最后的非零段，会从末尾向左裁掉 shrink_px 像素
    """
    mask2 = mask2.copy()
    h, w = mask2.shape

    for y in range(h):
        row = mask2[y]
        nonzero_indices = np.where(row > 0)[0]

        if len(nonzero_indices) == 0:
            continue

        # 找出右端连续非0段的开始和结束
        end = nonzero_indices[-1]
        start = end
        for i in reversed(nonzero_indices[:-1]):
            if i == start - 1:
                start = i
            else:
                break

        # 从 end 向左收缩 shrink_px 个像素
        shrink_start = max(start, end - shrink_px + 1)
        mask2[y, shrink_start:end + 1] = 0

    return mask2

METHOD = "orb"
frame_idx = 0
while True:
    for _ in range(SKIP_FRAMES - 1):
        cap.read()  # 跳过非关键帧

    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += SKIP_FRAMES
    if METHOD == "orb":
        H = register_feature_based(frame, ref_frame)
        if H is None:
            print(f"第 {frame_idx} 帧跳过")
            frame_idx += 1
            continue
        # warped = cv2.warpPerspective(frame, H, (canvas_h, canvas_w))
    elif METHOD == "optical_flow":
        warped = register_optical_flow(frame, ref_frame)
    elif METHOD == "phase":
        warped = register_phase_correlation(frame, ref_frame)
    else:
        raise ValueError("未知方法")
    kp_cur, des_cur = orb.detectAndCompute(frame, None)
    if des_ref is None or des_cur is None:
        continue

    # 累计 H（相对于第一帧）
    H_total = H_total @ H
    H_warp = offset @ H_total
    blur_img = stitch_with_feather_blend(ref_frame, frame, H)

    # warp 当前帧到全景图
    warped = cv2.warpPerspective(frame, H_warp, (canvas_w, canvas_h))
    # 融合（简单 max 合并）
    mask1 = np.any(panorama_canvas == 0, axis=2)  # shape: (H, W)
    mask2 = np.any(warped != 0, axis=2)
    mask2 = shrink_mask2_rowwise(mask2, 5)
    # 两图交集区域（两者都为非黑）
    intersection_mask = np.logical_and(mask1, mask2)
    panorama_canvas[intersection_mask] = warped[intersection_mask]
    intersection_mask = intersection_mask.astype(np.uint8)
    cv2.imshow("blur_img", blur_img)
    # cv2.imshow("warped", warped)
    # cv2.imshow("panorama_canvas", panorama_canvas)
    cv2.waitKey(0)
    # 更新参考帧
    ref_frame = frame
    kp_ref, des_ref = kp_cur, des_cur

    print(f"[✓] 拼接第 {frame_idx} 帧成功")

cap.release()

# 自动裁剪非黑区域
def crop_valid_region(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(contours[0])
    return img[y:y+h, x:x+w]

final_panorama = crop_valid_region(panorama_canvas)
cv2.imwrite(SAVE_PATH, final_panorama)
print(f"✅ 全景图保存成功：{SAVE_PATH}")
