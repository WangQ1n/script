import os
import cv2
import numpy as np
import time
import shutil
# 设置包含多个文件夹的根路径
root_dir = '/media/crrcdt123/glam/crrc/data/su8/2door/0822/fake/'  # 替换为你的路径

# 获取所有子文件夹
subfolders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

index = 0
while 0 <= index < len(subfolders):
    folder = subfolders[index]
    name = folder.split("/")[-1]
    # if name < "20000":
    #     index += 1
    #     continue
    img_paths = [os.path.join(folder, f"{i}.jpg") for i in range(3)]

    # 检查图片是否存在
    if not all(os.path.exists(p) for p in img_paths):
        print(f"跳过文件夹（缺图）: {folder}")
        index += 1  # 5291
        continue

    # 读取并统一尺寸
    images = [cv2.imread(p) for p in img_paths]
    h_min = min(img.shape[0] for img in images)
    w_min = min(img.shape[1] for img in images)
    images = [cv2.resize(img, (w_min, h_min)) for img in images]


    # 合并显示
    combined = cv2.hconcat(images)
    cv2.imshow('Images', combined)
    print(f"显示文件夹: {folder} | ← 上一个 | → 下一个 | d 删除 | Esc 退出")

    key = cv2.waitKey(0)

    if key == 27:  # ESC
        break
    elif key == ord('d'):
        shutil.rmtree(folder)
        print(f"已删除文件夹: {folder}")
        subfolders.pop(index)
        # 不增加 index，以显示下一个当前位置信息
    elif key == 81:  # ← Left Arrow
        index = max(index - 1, 0)
    elif key == 83 or key != -1:  # → Right Arrow 或任意其他键
        index += 1

cv2.destroyAllWindows()
