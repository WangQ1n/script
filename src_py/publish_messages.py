import rclpy
import time
import threading
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import PointCloud2, Imu, PointField
from livox_ros_driver2.msg import CustomMsg
from rclpy.serialization import deserialize_message
from rclpy.node import Node
import numpy as np
import struct

class BagMessagePublisher(Node):
    def __init__(self):
        super().__init__('bag_message_publisher')
        self.lidar_publisher_ = self.create_publisher(PointCloud2, '/C6/lidar', 10)
        self.imu_publisher_ = self.create_publisher(Imu, '/C6/imu', 10)

        # bag 文件路径
        self.bag_file_path = '/media/crrcdt123/glam/removter/rosbag/rosbag2_2025_01_02-15_29_37/'  # 替换为你的 bag 文件路径
        self.lidar_topic = '/C6/lidar'  # 替换为 lidar 点云话题名
        self.imu_topic = '/C6/imu'  # 替换为 IMU 话题名
        self.lidar_times = []
        self.imu_times = []
        self.lidar_msgs = []
        self.imu_msgs = []
        # self.lidar_times, self.imu_times, self.lidar_msgs, self.imu_msgs = self.get_timestamps_from_bag()

    def get_timestamps_from_bag(self):
        # lidar_times = []
        # imu_times = []
        # lidar_msgs = []
        # imu_msgs = []

        # 设置存储和转换选项
        storage_options = StorageOptions(uri=self.bag_file_path, storage_id='sqlite3')
        converter_options = ConverterOptions()

        # 创建读取器
        reader = SequentialReader()
        reader.open(storage_options, converter_options)

        # 读取 bag 包中的每条消息
        while reader.has_next():
            topic, msg_bytes, timestamp = reader.read_next()

            if topic == self.lidar_topic:
                # 解码字节流为 ROS 消息
                msg = deserialize_message(msg_bytes, PointCloud2)
                lidar_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1.e9  # 转换为秒
                self.lidar_times.append(lidar_time)
                self.lidar_msgs.append(msg)

            elif topic == self.imu_topic:
                msg = deserialize_message(msg_bytes, Imu)
                imu_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1.e9  # 转换为秒
                self.imu_times.append(imu_time)
                self.imu_msgs.append(msg)

    def publish_lidar_messages(self):
        # 发布 lidar 消息
        i = 1
        while True: 
            if i < len(self.lidar_msgs):
                msg = self.lidar_msgs[i]
                msg.header.stamp.sec = int(self.lidar_times[i])
                msg.header.stamp.nanosec = int((self.lidar_times[i] - int(self.lidar_times[i])) * 1e9)
                time_interval = self.lidar_times[i] - self.lidar_times[i-1]
                self.get_logger().info(f"Publishing lidar message with timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
                self.lidar_publisher_.publish(msg)
                i += 1
                time.sleep(time_interval)
    
    def publish_lidar_messages2(self):
        # 发布 lidar 消息
        idx = 1
        while True: 
            if idx < len(self.lidar_msgs):
                msg = self.lidar_msgs[idx]
                cloud_msg = PointCloud2()
                cloud_msg.header = msg.header  # 传递原始消息的头部

                # 设置 PointCloud2 的字段
                cloud_msg.height = 1  # 设定点云为一维
                cloud_msg.width = len(msg.points)  # 点的数量

                # 设置 PointCloud2 字段
                cloud_msg.fields = [
                    # 位置字段
                    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                    # 强度字段
                    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
                    # 标签字段
                    PointField(name="tag", offset=16, datatype=PointField.UINT8, count=1),
                    # 激光线编号字段
                    PointField(name="line", offset=17, datatype=PointField.UINT8, count=1),
                    # 时间戳字段
                    PointField(name="timestamp", offset=18, datatype=PointField.FLOAT64, count=1),
                ]

                cloud_msg.is_bigendian = False
                cloud_msg.point_step = 32  # 每个点的大小（4 字节 * 3 + 4 字节 * 1 + 1 字节 * 2 + 8 字节 * 1 = 26 字节）
                cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
                cloud_msg.is_dense = True

                # 将数据填充到 PointCloud2 消息中
                points = []

                for point in msg.points:
                    packed_point = struct.pack(
                        'ffffBBd',  # 对应x, y, z, intensity, tag, line, timestamp的格式
                        point.x,
                        point.y,
                        point.z,
                        point.reflectivity,
                        point.tag,
                        point.line,
                        point.offset_time
                    )
                    points.append(packed_point)
        
                # 将字节数据合并为一个大字节流
                data = b''.join(points)

                cloud_msg.data = data
                cloud_msg.header.stamp.sec = int(self.lidar_times[idx])
                cloud_msg.header.stamp.nanosec = int((self.lidar_times[idx] - int(self.lidar_times[idx])) * 1e9)
                cloud_msg.header.frame_id = 'base_link'
                time_interval = self.lidar_times[idx] - self.lidar_times[idx-1]
                self.get_logger().info(f"Publishing lidar message with timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
                self.lidar_publisher_.publish(cloud_msg)
                idx += 1
                time.sleep(time_interval)

    def publish_imu_messages(self):
        i = 1
        while True:
            if i < len(self.imu_msgs):
                msg = self.imu_msgs[i]
                msg.header.stamp.sec = int(self.imu_times[i])
                msg.header.stamp.nanosec = int((self.imu_times[i] - int(self.imu_times[i])) * 1e9)
                msg.header.frame_id = 'base_link'
                time_interval = self.imu_times[i] - self.imu_times[i-1]
                # self.get_logger().info(f"Publishing imu message with timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
                self.imu_publisher_.publish(msg)
                i += 1
                time.sleep(time_interval)

    def publish_imu_messages2(self):
        # 发布 imu 消息
        idx = 1
        while True:
            if idx < len(self.imu_msgs):
                msg = self.imu_msgs[idx]
                msg.header.stamp.sec = int(self.imu_times[idx])
                msg.header.stamp.nanosec = int((self.imu_times[idx] - int(self.imu_times[idx])) * 1e9)
                
                time_interval = self.imu_times[idx] - self.imu_times[idx-1]
                # self.get_logger().info(f"Publishing imu message with timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
                self.imu_publisher_.publish(msg)
                idx += 1
                time.sleep(time_interval)

    def start_publishing(self):
        # 启动两个线程分别处理 lidar 和 imu 消息
        load_thread = threading.Thread(target=self.get_timestamps_from_bag)
        lidar_thread = threading.Thread(target=self.publish_lidar_messages)
        imu_thread = threading.Thread(target=self.publish_imu_messages)

        load_thread.start()
        time.sleep(10)
        lidar_thread.start()
        imu_thread.start()

        # 等待线程完成
        load_thread.join()
        lidar_thread.join()
        imu_thread.join()

def main(args=None):
    rclpy.init(args=args)
    node = BagMessagePublisher()
    node.start_publishing()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
