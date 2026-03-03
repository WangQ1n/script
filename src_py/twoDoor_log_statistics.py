import os
import re
import shutil
from datetime import datetime

# 统计检测到的目标与总数量
# 配置参数
log_file = "/media/crrcdt123/glam/crrc/data/su8/2door/082810-20250630/日志/anti_clamp.log"  # 日志文件路径
keyword = "detected object"     # 需要提取的关键字段
image_folder = "/media/crrcdt123/glam/crrc/data/su8/2door/082810-20250630/pictures"  # 存放图片的目录
output_folder = "/home/crrcdt123/git/script/2door_log_check_result/082810/"  # 输出目录
timestamp_pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"  # 示例时间戳格式：2024-04-30_12-30-59

os.makedirs(output_folder, exist_ok=True)

def extract_timestamp_from_filename(filename):
    try:
        parts = filename.split("-")
        if len(parts) < 2:
            return None
        datetime_str = f"{parts[2]}-{parts[3].split('.')[0][:6]}"
        return datetime.strptime(datetime_str, "%Y%m%d-%H%M%S").timestamp()
    except Exception:
        return None
    

# 日志中的时间戳是纯秒数字
timestamp_pattern = r"\b\d{10}\b"  # 匹配10位数字
camera_pattern = r"cam\d{3,4}"
with open(log_file, 'r') as f:
    lines = f.readlines()

detect_count = 0
count = 0

for i, line in enumerate(lines):
    if "start detection" in line:
        detect_count += 1

    if keyword not in line:
        continue

    ts_match = re.search(timestamp_pattern, line)
    cam_match = re.search(camera_pattern, line)

    if not ts_match or not cam_match:
        continue

    target_ts = int(ts_match.group())
    cam_id = cam_match.group()

    # 获取该相机下所有图像
    images = [f for f in os.listdir(image_folder) if cam_id in f and f.endswith(".jpg")]
    if not images:
        continue

    # 提取图像时间戳并计算差值
    image_times = []
    for img in images:
        ts = extract_timestamp_from_filename(img)
        if ts:
            image_times.append((abs(ts - target_ts), img))

    # 选取时间差最小的两个图像
    image_times.sort(key=lambda x: x[0])
    top3_images = [img for diff, img in image_times[:3] if diff < 60]

    if not top3_images:
        continue

    # 拷贝图片
    for img in top3_images:
        shutil.copy(os.path.join(image_folder, img), os.path.join(output_folder, str(count) + "_" + img))

    print(f"[✓] 匹配成功（第{i+1}行）: {top3_images}")
    count += 1



print(f"\n detect: {detect_count}, abnormal: {count}, ration {(1 - count / (detect_count * 24.0))*100}%")