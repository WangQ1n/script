import cv2
import numpy as np

def roi_line_fitting(img):
    roi_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # print(np.shape(roi_img))
    # roi_img = cv2.resize(roi_img, (480, 270))
    # print(np.shape(roi_img))

    show_img = np.zeros((int(np.shape(roi_img)[0]/2), int(np.shape(roi_img)[1]/2), int(np.shape(roi_img)[2])), np.uint8) # 生成一个空灰度图像
    pointlist_l = []
    pointlist_r = []

    # TODO::-> numpy
    # left points
    for x in range(roi_img.shape[0]):
        for y in range(roi_img.shape[1]):
            px = roi_img[x, y, 0]
            if(px > 0):
                pointlist_l.append((y,x))
                break
    # right points
    for x in range(roi_img.shape[0]):
        for y in reversed(range(roi_img.shape[1])):
            px = roi_img[x, y, 0]
            if(px > 0):
                pointlist_r.append((y,x))
                break

    worning_distance = 20
    if len(pointlist_l) > 15 and len(pointlist_r) > 15:
        # ****************左侧:过滤点上下段********
        pointlist_l_n = np.array(pointlist_l)  # pointlist_left_np.array
        y_top = min(pointlist_l_n[:, 1])
        y_bottle = max(pointlist_l_n[:, 1])
        y_top_d = y_top + (y_bottle - y_top)*0.03
        y_bottle_d = y_bottle - (y_bottle - y_top)*0.07
        # print(y_top, y_bottle, y_top_d, y_bottle_d)
        points_middle_id = np.where((pointlist_l_n[:, 1] > y_top_d) & (pointlist_l_n[:, 1] < y_bottle_d))
        pointlist_l_n = pointlist_l_n[points_middle_id]

        # 左侧:拟合,转换x和y轴
        fl = np.polyfit(pointlist_l_n[:, 1], pointlist_l_n[:, 0], 7)  # 用3次多项式拟合
        pl = np.poly1d(fl)  # 求3次多项式表达式
        # print("左边多项式：", pl)
        lyy = pl(pointlist_l_n[:, 1])  # 拟合y值
        left_nh = np.vstack((lyy, pointlist_l_n[:, 1]))
        left_nh = np.transpose(left_nh)
        # print(left_nh.shape)

        # 左侧:worning line, 根据y获得等比的距离，让远处的worningdis更小
        left_worning_dis = np.arange(0, left_nh.shape[0], 1)
        left_worning_dis_0 = np.zeros(left_nh.shape[0])
        left_worning_dis = np.vstack((left_worning_dis, left_worning_dis_0))
        left_worning_dis = np.transpose(left_worning_dis)
        pointlist_l_n_worning = left_nh - (left_worning_dis*0.2 + np.array([[worning_distance, 0]]))
        # pointlist_l_n_worning = left_nh - np.array([[worning_distance, 0]])

        # *****************右侧:过滤点上下段**********
        pointlist_r_n = np.array(pointlist_r)
        y_top = min(pointlist_r_n[:, 1])
        y_bottle = max(pointlist_r_n[:, 1])
        y_top_d = y_top + (y_bottle - y_top)*0.03
        y_bottle_d = y_bottle - (y_bottle - y_top)*0.07
        # print(y_top, y_bottle, y_top_d, y_bottle_d)
        points_middle_id = np.where((pointlist_r_n[:, 1] > y_top_d) & (pointlist_r_n[:, 1] < y_bottle_d))
        pointlist_r_n = pointlist_r_n[points_middle_id]
        # print(pointlist_r_n)
        # 右侧:拟合
        fr = np.polyfit(pointlist_r_n[:, 1], pointlist_r_n[:, 0], 7)  # 用3次多项式拟合
        pr = np.poly1d(fr)  # 求3次多项式表达式
        # print("右边多项式：", pr)
        ryy = pr(pointlist_r_n[:, 1])  # 拟合y值
        right_nh = np.vstack((ryy, pointlist_r_n[:, 1]))
        right_nh = np.transpose(right_nh)
        # print(right_nh.shape)

        # 右侧:worning line
        right_worning_dis = np.arange(0, right_nh.shape[0], 1)
        right_worning_dis_0 = np.zeros(right_nh.shape[0])
        right_worning_dis = np.vstack((right_worning_dis, right_worning_dis_0))
        right_worning_dis = np.transpose(right_worning_dis)
        pointlist_r_n_worning = right_nh + (right_worning_dis*0.2 + np.array([[worning_distance, 0]]))
        # pointlist_r_n_worning = right_nh + np.array([[worning_distance, 0]])

        # 画线
        point_size = 1
        point_color_l = (0, 255, 255)  # BGR
        point_color_r = (0, 255, 255)  # BGR
        point_color_l_worning = (0, 0, 255)  # BGR
        point_color_r_worning = (0, 0, 255)  # BGR
        thickness = 2  # 可以为 0 、4、8
        # for point in pointlist_l_n:
        #     cv2.circle(roi_img, (int(point[0]), int(point[1])), point_size, point_color_l, thickness)
        # for point in pointlist_r_n:
        #     cv2.circle(roi_img, (int(point[0]), int(point[1])), point_size, point_color_r, thickness)
        for point in pointlist_l_n_worning:
            cv2.circle(roi_img, (int(point[0]), int(point[1])), point_size, point_color_l_worning, thickness)
        for point in pointlist_r_n_worning:
            cv2.circle(roi_img, (int(point[0]), int(point[1])), point_size, point_color_r_worning, thickness)
        # 画拟合曲线
        for i in range(left_nh.shape[0]):
            if (i+1 == left_nh.shape[0]):
                break
            cv2.line(roi_img, (int(left_nh[i][0]), int(left_nh[i][1])), (int(left_nh[i+1][0]), int(left_nh[i+1][1])), point_color_l, 2)
        for i in range(right_nh.shape[0]):
            if (i+1 == right_nh.shape[0]):
                break
            cv2.line(roi_img, (int(right_nh[i][0]), int(right_nh[i][1])), (int(right_nh[i+1][0]), int(right_nh[i+1][1])), point_color_r, 2)

    
    cv2.imshow('pic title', roi_img)

    return roi_img

def main():
    root = "/home/crrcdt123/pytorch_test/pytorch-deeplab-xception/traindata/train_voc/"
    mask_path = root + "/SegmentationClass/frame_001034++frame_001045.png"
    mask = cv2.imread(mask_path)
    mask = mask[:int(mask.shape[0]/2), ...]
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            px = mask[x, y, 2]
            if(px > 0):
                mask[x, y, 0] = 0
                mask[x, y, 1] = 255
                mask[x, y, 2] = 0
    roi = roi_line_fitting(mask)

    img_path = root + "/JPEGImages/frame_001034++frame_001045.jpg"
    img = cv2.imread(img_path)
    img = img[:int(img.shape[0]/2), ...]
    cv2.imshow("img", img)
    # img = cv2.resize(img, (480, 270)) 
    overlay = cv2.addWeighted(img, 0.7, roi, 0.3, 0)
    cv2.imshow("overlay", overlay)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()