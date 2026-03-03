import cv2
import os
from ultralytics import YOLO

# 参数设置
'''
/media/crrcdt123/glam/crrc/data/su8/video_raw/20240307/0022-20240307-114517.mp4
/media/crrcdt123/glam/运行素材/video_cam_short-20250210-055521-11-C1.mkv
'''
video_path = '/media/crrcdt123/glam/运行素材/video_cam_short-20250210-071521-19-C1.mkv'              # 输入视频路径
model_path = '/home/crrcdt123/git/ultralytics/runs/detect/train24/weights/best.pt'                  # 模型权重，可为 yolov8s.pt / yolov8n.pt / 自训.pt
output_dir = 'video_cam_short-20250210-071521-19-C1'                # 保存目录

os.makedirs(output_dir, exist_ok=True)

model = YOLO(model_path)  # 加载模型
cap = cv2.VideoCapture(video_path)

frame_idx = 0
save_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    # 如果检测到目标
    if len(results.boxes) > 0:
        annotated_frame = results.plot()  # 带框可视化图
        save_path = os.path.join(output_dir, f'frame_{frame_idx:05d}.jpg')
        cv2.imwrite(save_path, annotated_frame)
        print(f"✅ Saved: {save_path}")
        save_count += 1

    frame_idx += 1

cap.release()
print(f"\n🎯 Done. Total saved frames: {save_count}")
