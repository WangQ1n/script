# -*- coding: utf-8 -*-
import os
import cv2

def main():
    """
    抽帧获取图像
    """
    root = "/media/crrcdt123/glam1/crrc/datasets/桥林/"
    name = "video_cam_short-20240106-103345-2-C6.mkv"
    video_path = os.path.join(root, name)
    save_dir = os.path.join("/media/crrcdt123/glam1/crrc/datasets/桥林/", "images")
    save_path = os.path.join(save_dir, "%s_%s.jpg")
    cap = cv2.VideoCapture(video_path)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    # fps = cap.get(cv2.CAP_PROP_FPS)
    video_frame = 0
    while True:
        ret_val, frame = cap.read()
        if ret_val is False:
            break
        if video_frame % 25 == 0:
            tmp = save_path % (name.replace(".mp4", ""), video_frame)
            cv2.imwrite(save_path % (name.replace(".mp4", ""), video_frame), frame)
            cv2.imshow("img", frame)
        video_frame = video_frame + 1
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

if __name__ == '__main__':
    main()
