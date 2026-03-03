import os
import cv2
import glob
import numpy as np

def draw_labels(image, json_path):
    with open(json_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 7:  # 分割标签至少需要: class_id + 多个点(x,y)
            continue
        
        points = []
        
        # 解析多边形点（归一化坐标 -> 绝对坐标）
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                x = float(parts[i]) * 224
                y = float(parts[i + 1]) * 224
                points.append([x, y])
        
        if len(points) < 3:
            continue
        
        points = np.array(points, dtype=np.int32)
        cv2.polylines(image, [points], True, (0, 255, 0), 1)
    return image


def main():
    # 设置包含多个文件夹的根路径
    root_dir = '/media/crrcdt123/glam/crrc/data/su8/2door/0806/yolo_unblue/'  # 替换为你的路径
    image_dir = os.path.join(root_dir, 'images', 'train')
    label_dir = os.path.join(root_dir, 'labels')
    all_images = []
    full_pattern = os.path.join(image_dir, '**/*.jpg')
    all_images = glob.glob(full_pattern, recursive=True)
    # # 获取所有子文件夹
    # for item in os.listdir(root_dir):
    #     full_path = os.path.join(root_dir, item)
    #     all_images.append(full_path)
    index = 0
    find = True
    while 0 <= index < len(all_images):
        image = all_images[index]
        # 05181.jpg next
        # if not find:
        #     if "FP" in image:
        #         find = True
        #     else:
        #         index += 1
        #         continue
        # if "FP" not in image:
        #     index += 1
        #     continue
        if "train" in image:
            image2 = image.replace("train", "train2")
            txt_path = image.replace("images/train", "labels/train").replace(".jpg", ".txt")
        else:
            image2 = image.replace("val", "val2")
            txt_path = image.replace("images/val", "labels/val").replace(".jpg", ".txt")
        img_paths = [image2, image]
        check_paths = [image2, image, txt_path]
        if not all(os.path.exists(p) for p in check_paths):
            print(f"跳过文件夹（缺图）: {image}")
            index += 1
            continue
        # 读取并统一尺寸
        images = [cv2.imread(p) for p in img_paths]
        h_min = min(img.shape[0] for img in images)
        w_min = min(img.shape[1] for img in images)
        images = [cv2.resize(img, (w_min, h_min)) for img in images]
        images[0] = draw_labels(image=images[0], json_path=txt_path)
        # 合并显示
        combined = cv2.hconcat(images)
        cv2.imshow('Images', combined)
        print(f"显示文件夹: {image} | ← 上一个 | → 下一个 | d 删除 | Esc 退出")

        key = cv2.waitKey(0)

        if key == 27:  # ESC
            break
        elif key == ord('d'):
            os.remove(image)
            os.remove(image2)
            os.remove(txt_path)
            print(f"已删除文件夹: {image}")
            all_images.pop(index)
            # 不增加 index，以显示下一个当前位置信息
        elif key == 81:  # ← Left Arrow
            index = max(index - 1, 0)
        elif key == 83 or key != -1:  # → Right Arrow 或任意其他键
            index += 1
        else:
            index += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()