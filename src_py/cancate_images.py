import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread('/home/crrcdt123/git/script/c.jpg')  # 上一帧
img2 = cv2.imread('/home/crrcdt123/git/script/d.jpg')  # 当前帧

# 缩小方便显示
img1 = cv2.resize(img1, (768, 1024))
img2 = cv2.resize(img2, (768, 1024))

def phase_correlation_shift(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    shift, response = cv2.phaseCorrelate(gray1, gray2)
    dx, dy = shift
    print(f"[Phase Correlation] Estimated shift: dx={dx:.2f}, dy={dy:.2f}")
    return int(round(dx)), int(round(dy))

# dx, dy = phase_correlation_shift(img1, img2)
# aligned = np.roll(img2, shift=-dx, axis=1)

def sift_ransac_align(img1, img2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    aligned = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

    print(f"[SIFT] Matches used: {len(good)}")
    return aligned

aligned = sift_ransac_align(img1, img2)

def ecc_alignment(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    warp_matrix = np.eye(2, 3, dtype=np.float32)  # 仿射变换
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)

    try:
        cc, warp_matrix = cv2.findTransformECC(gray1, gray2, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        aligned = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        print(f"[ECC] Converged with correlation = {cc:.4f}")
        return aligned
    except cv2.error as e:
        print(f"[ECC] Failed: {e}")
        return img2  # fallback

def show_alignment(img1, aligned):
    overlay = cv2.addWeighted(img1, 0.5, aligned, 0.5, 0)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlay After Alignment")
    plt.axis('off')
    plt.show()

# show_alignment(img1, aligned)

def extract_frames_from_video(video_path, interval=10, max_frames=None):
    """从视频中每隔 interval 帧提取一帧"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(frame)
        count += 1
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    print(f"共提取 {len(frames)} 帧")
    return frames

def split_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def stitch_images(images, mode=cv2.Stitcher_SCANS):
    stitcher = cv2.Stitcher_create()
    status, pano = stitcher.stitch(images)
    if status == cv2.Stitcher_OK:
        return pano
    else:
        print(f"拼接失败，状态码：{status}")
        return None

def save_image(name, img):
    cv2.imwrite(name, img)
    print(f"已保存: {name}")


# === 主逻辑 ===
stitcher = cv2.Stitcher_create()
video_path = "/home/crrcdt123/git/script/keyboard.mp4"   # ← 替换为你的视频路径
frame_interval = 20                      # 每隔几帧取一帧
group_size = 10                           # 每组拼接多少张
frame_count = 1
cap = cv2.VideoCapture(video_path)
ret, panorama = cap.read()
while True:
    ret, frame = cap.read()
    frame_count += 1
    if frame_count % frame_interval != 0:
        continue
    if not ret:
        break

    # 尝试将 panorama 和当前 frame 拼接
    status, result = stitcher.stitch([panorama, frame])

    if status == cv2.Stitcher_OK:
        panorama = result
        print(f"[✓] 拼接第 {frame_count} 帧成功")
        cv2.imshow("Final Panorama", panorama)
        cv2.waitKey(0)
    else:
        print(f"[×] 第 {frame_count} 帧拼接失败，状态码：{status}")

if panorama is not None:
    save_image("final_panorama.jpg", panorama)
    cv2.imshow("Final Panorama", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("最终拼接失败。")
