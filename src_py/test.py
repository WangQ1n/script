import cv2
print(cv2.__version__)
# 读取视频文件
video = cv2.VideoCapture('/media/crrcdt123/glam/crrc/datasets/s8/video_raw/0022-20240307-111214.mp4')	# 参数为视频文件地址，若是数字表示摄像头编号。
'''
参数为字符串，表示输入的视频文件的地址及文件名
参数为数字，表示摄像头编号，默认为-1.即随机选取一个摄像头
'''

# 创建写视频器
video_writer = cv2.VideoWriter(filename='./output.mp4', 				# 保存路径文件名
							   apiPreference=cv2.CAP_FFMPEG,			# 后端
							   fourcc=cv2.VideoWriter_fourcc(*'hvc1'),	# 视频编解码器
							   fps=25,									# 视频帧率
							   frameSize=(1920,1080),					# 视频帧尺寸(W, H)
							   isColor=True								# 彩色图像或黑白图像
							   )
'''
apiPreference：cv2.CAP_FFMPEG 或者 cv2.CAP_GSTREAMER，此参数是3.x版本的opencv才有的
fourcc：cv2.VideoWriter_fourcc('M', 'P', '4', 'V') 或 fourcc=cv2.VideoWriter_fourcc(*'mp4v')
		支持类型：MP4V / X264 / I420 / PIMI / XVID / THEO / FLV1
'''							   
while video_writer.isOpened():
  # 判断视频是否是打开状态
  while video.isOpened():
    # 读取一帧
    ret, frame = video.read() # ret是bool类型，表示是否读取成功；frame为获取的帧图像

    if ret:
      # 写入一帧
      video_writer.write(frame)
      # 播放视频
      cv2.imshow('frame', frame)
      cv2.waitKey(1) # 通过设置等待时间改变播放速度
    else:
      break

# 释放
video.release()
video_writer.release()	# 不释放会无法完成写视频，类似文件写完后的close()
