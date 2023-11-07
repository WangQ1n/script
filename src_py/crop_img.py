# -*- coding: utf-8 -*-
import os
import json
import cv2 as cv

def main():
    """
    抽帧获取图像
    """
    cam_name = "468"
    x, y, width, height = 0, 0, 0, 0
    if cam_name == "357":
        x, y, width, height = 500, 200, 940, 55
    elif cam_name == "157":
        x, y, width, height = 470, 230, 920, 55
    elif cam_name == "413":
        x, y, width, height = 370, 226, 940, 55
    elif cam_name == "424":
        x, y, width, height = 550, 220, 940, 55
    elif cam_name == "468":
        x, y, width, height = 770, 180, 870, 55
    cam_name = "cam" + cam_name
    src_dir = "/home/crrcdt123/datasets2/二门防夹数据/车间"
    save_dir = "/home/crrcdt123/datasets2/二门防夹数据/车间背景/" + cam_name
    if os.path.exists(save_dir) is False:
        os.system("mkdir " + save_dir)
    all_path = os.listdir(src_dir)
    idx = 0
    for path in all_path:
        if cam_name in path:
            if "Start" in path or "End" in path:
                img = cv.imread(os.path.join(src_dir, path))
                crop_img = img[y:y+height, x:x+width]
                cv.imwrite(os.path.join(save_dir, cam_name + "_" + str(idx)+".jpg"), crop_img)
                idx+=1

if __name__ == '__main__':
    main()
