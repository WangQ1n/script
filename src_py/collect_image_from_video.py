# -*- coding: utf-8 -*-
import os
import cv2


def main():
    """
    抽帧获取图像
    """
    root = "/media/crrcdt123/glam/crrc/data/su7/20260206/video_obj/"
    # name = "event_video_cam_long-20251013-122341-C1.mkv"
    save_dir = os.path.join(root, "images")
    save_path = os.path.join(save_dir, "%s_%s.jpg")
    for filename in os.listdir(root):
        if filename.lower().endswith(".mkv"):
            video_path = os.path.join(root, filename)
            print("process ", video_path)
            cap = cv2.VideoCapture(video_path)
            video_frame = 0
            while True:
                ret_val, frame = cap.read()
                if ret_val is False:
                    break
                if video_frame % 10 == 0:
                    cv2.imwrite(save_path %
                                (filename.replace(".mkv", ""), video_frame), frame)
                    frame = cv2.resize(frame, (960, 540))
                    cv2.imshow("img", frame)
                video_frame = video_frame + 1
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break


if __name__ == '__main__':
    main()
