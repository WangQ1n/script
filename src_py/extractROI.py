import cv2
print(cv2.__version__)
assert hasattr(cv2, 'dnn_superres')
print(cv2.getBuildInformation())
class MultiScaleLetterbox:
    def __init__(self, sizes=[512, 640, 768]):
        self.sizes = sizes
        self.model = cv2.dnn_superres.DnnSuperResImpl_create()  # 1. 创建实例

        # 2. 读取模型（检查返回值）
        # ret = self.model.readModel("/home/crrcdt123/git/script/ESPCN_x3.pb")  # 正确应返回True
        ret = self.model.readModel("/home/crrcdt123/git/script/EDSR_3x.pb")  # 正确应返回True
        self.model.setModel("edsr", 3)
        print(f"模型读取返回值: {ret}")
    
    def __call__(self, img):
        # 选择性超分增强
        padded = self.model.upsample(img)
        
        return padded  # 返回多尺度结果供模型融合
    

upsample = MultiScaleLetterbox(640)

def extract_roi_from_video(input_video_path, output_video_path, roi):
    """
    从视频中提取ROI区域并保存为新视频
    
    参数:
        input_video_path: 输入视频路径
        output_video_path: 输出视频路径
        roi: 感兴趣区域 (x, y, width, height)
    """
    # 打开视频文件
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return
    
    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 检查ROI是否有效
    x, y, w, h = roi
    if (x + w > original_width) or (y + h > original_height):
        print("ROI区域超出视频范围")
        return
    
    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID' 对应.avi格式
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    
    print(f"开始处理视频: {input_video_path}")
    print(f"视频信息: {original_width}x{original_height}, {fps:.2f} FPS, {frame_count}帧")
    print(f"提取ROI区域: x={x}, y={y}, width={w}, height={h}")
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num > 0*60*5:
        # 提取ROI区域
            roi_frame = frame[y:y+h, x:x+w]
        # roi_frame2 = cv2.resize(roi_frame, (900, 900))
        # up_frame = upsample(roi_frame)

        # cv2.imshow("up", up_frame)
        # cv2.imshow("src", roi_frame)
        # cv2.imshow("resize", roi_frame2)
        # cv2.waitKey(0)
        # 写入输出视频
            out.write(roi_frame)
        
        frame_num += 1
        if frame_num % 100 == 0:
            print(f"已处理 {frame_num}/{frame_count} 帧")
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"处理完成! 输出视频已保存到: {output_video_path}")


def extract_roi_from_image(image_path, roi):
    img = cv2.imread(image_path)
    size = img.shape
    # 检查ROI是否有效
    original_width = size[0]
    original_height = size[1]
    x, y, w, h = roi
    if (x + w > original_width) or (y + h > original_height):
        print("ROI区域超出视频范围")
        return
    # 提取ROI区域
    roi_frame = img[y:y+h, x:x+w]
    roi_frame2 = cv2.resize(roi_frame, (900, 900))
    up_frame = upsample(roi_frame)

    cv2.imshow("up", up_frame)
    cv2.imshow("src", roi_frame)
    cv2.imshow("resize", roi_frame2)
    cv2.waitKey(0)
    
# 使用示例
if __name__ == "__main__":
    input_video = "/media/crrcdt123/glam/crrc/data/su8/video_raw/20240307/0022-20240307-114517.mp4"  # 输入视频路径
    output_video = "/media/crrcdt123/glam/crrc/data/su8/video_raw/20240307/output_roi.mp4"  # 输出视频路径
    image_path = "/media/crrcdt123/glam/壁纸/海边书店风景.jpg"
    # 定义ROI区域 (800, 200, 250, 250)
    # 示例: 从视频中心提取300x300的区域
    roi_region = (600, 50, 640, 640)  # 假设原视频是1280x720
    
    extract_roi_from_video(input_video, output_video, roi_region)
    # extract_roi_from_image()