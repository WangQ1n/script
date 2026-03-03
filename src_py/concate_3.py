import cv2
import numpy as np

class VideoPanoramaStitcher:
    def __init__(self):
        # 特征检测与匹配配置
        self.detector = cv2.ORB_create(nfeatures=5000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 混合器配置
        self.blender = cv2.detail_FeatherBlender(sharpness=0.015)
        
        # 全景图状态
        self.panorama = None
        self.panorama_mask = None
        self.corners = []      # 各图像在全景图中的左上角坐标
        self.sizes = []        # 各图像的尺寸
        
        # 帧处理状态
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        self.frame_count = 0
        self.skip_frames = 3    # 处理间隔帧数
        
        # 运动检测阈值
        self.min_matches = 15
        self.max_match_dist = 30.0
        
    def process_frame(self, frame):
        self.frame_count += 1
        
        # 跳过部分帧以提高性能
        if self.frame_count % self.skip_frames != 0:
            return self.panorama
        
        # 调整帧大小 (提高处理速度)
        frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
        
        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测特征点
        kp, des = self.detector.detectAndCompute(gray, None)
        
        # 第一帧处理
        if self.prev_frame is None:
            self._initialize_panorama(frame, kp, des)
            return self.panorama
        
        # 检查特征点数量
        if des is None or len(kp) < self.min_matches:
            print(f"特征点不足 ({len(kp) if des is not None else 0}/{self.min_matches})，跳过此帧")
            return self.panorama
        
        # 特征匹配
        matches = self.matcher.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # 筛选优质匹配
        good_matches = [m for m in matches if m.distance < self.max_match_dist]
        
        if len(good_matches) < self.min_matches:
            print(f"优质匹配不足 ({len(good_matches)}/{self.min_matches})，跳过此帧")
            return self.panorama
        
        # 计算单应性矩阵
        H = self._calculate_homography(good_matches, kp)
        if H is None:
            return self.panorama
        
        # 更新全景图
        self._update_panorama(frame, H)
        
        # 更新前一帧信息
        self.prev_frame = frame.copy()
        self.prev_kp = kp
        self.prev_des = des
        
        return self.panorama
    
    def _initialize_panorama(self, frame, kp, des):
        """初始化全景图"""
        self.prev_frame = frame.copy()
        self.prev_kp = kp
        self.prev_des = des
        
        self.panorama = frame.copy()
        self.panorama_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        self.corners.append((0, 0))
        self.sizes.append((frame.shape[1], frame.shape[0]))
        
        print("全景图初始化完成")
    
    def _calculate_homography(self, matches, kp):
        """计算单应性矩阵"""
        src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            print("无法计算单应性矩阵，跳过此帧")
            return None
        
        # 检查单应性矩阵合理性
        if abs(H[0, 0] - 1.0) > 0.5 or abs(H[1, 1] - 1.0) > 0.5:
            print("单应性矩阵变形过大，可能匹配错误")
            return None
            
        return H
    
    def _update_panorama(self, frame, H):
        """更新全景图"""
        h, w = frame.shape[:2]
        
        # 计算当前帧在全景图中的位置
        corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        
        # 计算新全景图的边界
        new_x, new_y, new_w, new_h = self._calculate_new_canvas(warped_corners)
        
        # 调整所有图像的位置
        self._adjust_canvas(new_x, new_y, new_w, new_h)
        
        # 变换当前帧到全景图坐标系
        warped = cv2.warpPerspective(frame, H, (new_w, new_h))
        warped_mask = cv2.warpPerspective(np.ones((h, w), dtype=np.uint8)*255, 
                                        H, (new_w, new_h))
        
        # 准备混合器
        self.blender.prepare((0, 0, new_w, new_h))
        
        # 添加全景图和当前帧到混合器
        self.blender.feed(self.panorama.astype(np.int16), self.panorama_mask, (0, 0))
        self.blender.feed(warped.astype(np.int16), warped_mask, (0, 0))
        
        # 执行混合
        self.panorama, self.panorama_mask = self.blender.blend(None, None)
        self.panorama = self.panorama.astype(np.uint8)
        
        # 更新当前帧的位置信息
        curr_corner = (int(-new_x), int(-new_y))
        self.corners.append(curr_corner)
        self.sizes.append((w, h))
        
        print(f"全景图更新: 尺寸 {new_w}x{new_h}, 当前帧位置 {curr_corner}")
    
    def _calculate_new_canvas(self, warped_corners):
        """计算新的画布尺寸"""
        # 获取当前帧变换后的边界
        x, y, w, h = cv2.boundingRect(warped_corners)
        
        # 如果没有全景图，直接使用当前帧的边界
        if not self.corners:
            return x, y, w, h
        
        # 计算所有图像角落的边界
        all_points = [np.array([(c[0], c[1]) for c in self.corners])]
        all_points.append(warped_corners.reshape(4, 2))
        all_points = np.concatenate(all_points)
        
        new_x = int(np.floor(min(all_points[:, 0])))
        new_y = int(np.floor(min(all_points[:, 1])))
        new_w = int(np.ceil(max(all_points[:, 0]))) - new_x
        new_h = int(np.ceil(max(all_points[:, 1]))) - new_y
        
        # 确保最小尺寸
        new_w = max(new_w, self.panorama.shape[1] if self.panorama is not None else 0)
        new_h = max(new_h, self.panorama.shape[0] if self.panorama is not None else 0)
        
        return new_x, new_y, new_w, new_h
    
    def _adjust_canvas(self, new_x, new_y, new_w, new_h):
        """调整画布和已有图像位置"""
        if self.panorama is None:
            self.panorama = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            self.panorama_mask = np.zeros((new_h, new_w), dtype=np.uint8)
            return
        
        # 创建新的画布
        new_panorama = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        new_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        
        # 计算偏移量
        offset_x = -new_x
        offset_y = -new_y
        
        # 将原全景图复制到新位置
        orig_x1 = self.corners[0][0] + offset_x
        orig_y1 = self.corners[0][1] + offset_y
        orig_x2 = orig_x1 + self.panorama.shape[1]
        orig_y2 = orig_y1 + self.panorama.shape[0]
        
        # 确保不越界
        orig_x1 = max(orig_x1, 0)
        orig_y1 = max(orig_y1, 0)
        orig_x2 = min(orig_x2, new_w)
        orig_y2 = min(orig_y2, new_h)
        
        # 计算源图像的有效区域
        src_x1 = max(0, -self.corners[0][0] - offset_x)
        src_y1 = max(0, -self.corners[0][1] - offset_y)
        src_x2 = min(self.panorama.shape[1], new_w - (self.corners[0][0] + offset_x))
        src_y2 = min(self.panorama.shape[0], new_h - (self.corners[0][1] + offset_y))
        
        if src_x1 < src_x2 and src_y1 < src_y2:
            new_panorama[orig_y1:orig_y2, orig_x1:orig_x2] = \
                self.panorama[src_y1:src_y2, src_x1:src_x2]
            new_mask[orig_y1:orig_y2, orig_x1:orig_x2] = \
                self.panorama_mask[src_y1:src_y2, src_x1:src_x2]
        
        # 更新全景图和角点
        self.panorama = new_panorama
        self.panorama_mask = new_mask
        
        # 更新所有角点位置
        for i in range(len(self.corners)):
            self.corners[i] = (self.corners[i][0] + offset_x, 
                              self.corners[i][1] + offset_y)

def main():
    # 初始化拼接器
    stitcher = VideoPanoramaStitcher()
    
    # 打开视频流
    cap = cv2.VideoCapture("/home/crrcdt123/git/script/keyboard.mp4")  # 使用摄像头
    # cap = cv2.VideoCapture('input.mp4')  # 使用视频文件
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("开始视频流全景拼接...")
    print("操作指南:")
    print(" - 缓慢移动摄像头扫描场景")
    print(" - 按 's' 保存当前全景图")
    print(" - 按 'q' 退出程序")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧
        panorama = stitcher.process_frame(frame)
        
        # 显示
        cv2.imshow("Camera", frame)
        if panorama is not None:
            cv2.imshow("Panorama", panorama)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and panorama is not None:
            cv2.imwrite("panorama_snapshot.jpg", panorama)
            print("全景图已保存为 panorama_snapshot.jpg")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    # 最终保存
    if stitcher.panorama is not None:
        cv2.imwrite("final_panorama.jpg", stitcher.panorama)
        print("最终全景图已保存为 final_panorama.jpg")

if __name__ == '__main__':
    main()