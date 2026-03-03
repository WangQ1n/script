import rclpy
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
import matplotlib.pyplot as plt
from datetime import timedelta
from sensor_msgs.msg import PointCloud2, Imu
from rclpy.serialization import deserialize_message

def read_bag_and_calculate_time_intervals(bag_file_path, topic_name):
    # 设置存储和转换选项
    storage_options = StorageOptions(uri=bag_file_path, storage_id='sqlite3')
    converter_options = ConverterOptions()

    # 创建读取器
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # 用于记录时间间隔
    time_intervals = []
    timestamp_intervals = []
    last_timestamp = None
    last_self_timestamp = None
    # 读取包中的每条消息
    while reader.has_next():
        topic, msg_bytes, timestamp = reader.read_next()
        if topic == topic_name:
            if "lidar" in topic:
                msg = deserialize_message(msg_bytes, PointCloud2)
            elif "imu" in topic:
                msg = deserialize_message(msg_bytes, Imu)
            current_time = timestamp / 1e9  # 将时间戳转换为秒
            message_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1.e9  # 转换为秒
            
            if last_timestamp is not None:
                # 计算时间间隔
                time_interval = current_time - last_timestamp
                time_intervals.append(time_interval)

                # 计算消息之间的时间戳间隔
                timestamp_interval = message_time - last_self_timestamp
                timestamp_intervals.append(timestamp_interval)
            last_timestamp = current_time
            last_self_timestamp = message_time
    return time_intervals, timestamp_intervals


def plot_time_intervals(time_intervals, timestamp_intervals, name):
# 创建折线图
    plt.figure(figsize=(10, 6))
    plt.plot(time_intervals, label="Recorded Timestamp", color='blue', linestyle='-', marker='o')
    plt.plot(timestamp_intervals, label="Message Timestamp", color='red', linestyle='-', marker='x')

    # 添加图表标题和标签
    plt.title(name)
    plt.xlabel("Message Index")
    plt.ylabel("Timestamp (seconds)")
    plt.legend()




def main():
    # 指定 bag 包路径和话题名
    bag_file_path = '/home/crrcdt123/git/fastlio2_ros2/rosbag/rosbag2_2025_01_02-15_29_37/'  # 替换为你的bag文件路径
    topic_name = '/C6/lidar'  # 替换为你要读取的话题名

    # 获取时间间隔数据
    time_intervals, timestamp_intervals = read_bag_and_calculate_time_intervals(bag_file_path, topic_name)
    imu_time_intervals, imu_timestamp_intervals = read_bag_and_calculate_time_intervals(bag_file_path, "/C6/imu")
    # 绘制时间间隔的折线图
    if time_intervals:
        plot_time_intervals(time_intervals, timestamp_intervals, "lidar")
        plot_time_intervals(imu_time_intervals, imu_timestamp_intervals, "imu")
    else:
        print(f"No messages found for topic '{topic_name}'")
    # 显示图表
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()



