import os
import json
import cv2
import numpy as np

# 设置图像和标注路径
image_dir = "/media/crrcdt123/glam/public_datasets/coco/images"       # 替换为图像文件夹路径
label_dir = "/media/crrcdt123/glam/public_datasets/coco/labelme"       # 替换为标注JSON文件夹路径
yolo_dir = "/media/crrcdt123/glam/public_datasets/coco/yolo"
image_exts = ['.jpg', '.png', '.jpeg']

# 获取图像文件名（不含路径）
image_list = sorted([
    f for f in os.listdir(image_dir)
    if os.path.splitext(f)[1].lower() in image_exts
])

idx = 0  # 当前索引

while 0 <= idx < len(image_list):
    image_name = image_list[idx]
    image_path = os.path.join(image_dir, image_name)
    label_name = os.path.splitext(image_name)[0] + ".json"
    label_path = os.path.join(label_dir, label_name)
    yolo_name = os.path.splitext(image_name)[0] + ".txt"
    yolo_path = os.path.join(yolo_dir, yolo_name)

    image = cv2.imread(image_path)
    if image is None:
        print(f"[跳过] 无法读取图像: {image_path}")
        idx += 1
        continue

    # 如果 JSON 存在，绘制标注
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            data = json.load(f)

        for shape in data.get("shapes", []):
            points = shape["points"]
            label = shape.get("label", "")
            pts = [(int(x), int(y)) for x, y in points]

            if len(pts) == 2:  # 矩形框
                cv2.rectangle(image, pts[0], pts[1], (0, 255, 0), 2)
            else:  # 多边形
                cv2.polylines(image, [np.array(pts, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

            # 标签文字
            cv2.putText(image, label, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 显示图像
    # image = cv2.resize(image, (1536, 864))
    cv2.imshow("Labelme Viewer", image)
    key = cv2.waitKey(0)

    if key == ord('q') or key == 27:  # 退出
        break
    elif key == ord('d'):  # 删除图像 + JSON
        print(f"[删除] {image_name} , {label_name}, {yolo_name}")
        os.remove(image_path)
        if os.path.exists(label_path):
            os.remove(label_path)
        if os.path.exists(yolo_path):
            os.remove(yolo_path)
        image_list.pop(idx)
        if idx >= len(image_list):
            idx = len(image_list) - 1
    elif key == 81:  # ← 左键
        idx = max(0, idx - 1)
    elif key == 83:  # → 右键
        idx = min(len(image_list) - 1, idx + 1)
    else:
        idx += 1

cv2.destroyAllWindows()
