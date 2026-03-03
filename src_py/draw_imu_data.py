import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Imu
import rosbag2_py
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R


def read_imu_from_bag(bag_path, topic_name):
    # 初始化存储数据
    timestamps = []
    accel_x, accel_y, accel_z = [], [], []
    gyro_x, gyro_y, gyro_z = [], [], []

    # 设置 bag 文件存储选项
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr', output_serialization_format='cdr'
    )

    # 打开 bag 文件
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # 获取所有话题信息
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    # 确保 IMU 数据类型匹配
    imu_type = type_map.get(topic_name)
    if imu_type != 'sensor_msgs/msg/Imu':
        raise ValueError(f"Topic {topic_name} is not of type sensor_msgs/msg/Imu.")

    imu_msg_type = get_message(imu_type)

    # 读取消息
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic == topic_name:
            imu_msg = deserialize_message(data, imu_msg_type)
            message_time = imu_msg.header.stamp.sec + imu_msg.header.stamp.nanosec / 1.e9  # 转换为秒
            # 提取时间戳和数据
            timestamps.append(message_time)  # 转换为秒
            accel_x.append(imu_msg.linear_acceleration.x)
            accel_y.append(imu_msg.linear_acceleration.y)
            accel_z.append(imu_msg.linear_acceleration.z)
            gyro_x.append(imu_msg.angular_velocity.x)
            gyro_y.append(imu_msg.angular_velocity.y)
            gyro_z.append(imu_msg.angular_velocity.z)

    # 使用 pandas 创建表格
    df = pd.DataFrame({
        'Timestamp (s)': timestamps,
        'Accel X (m/s²)': accel_x,
        'Accel Y (m/s²)': accel_y,
        'Accel Z (m/s²)': accel_z,
        'Gyro X (deg/s)': gyro_x,
        'Gyro Y (deg/s)': gyro_y,
        'Gyro Z (deg/s)': gyro_z
    })

    return df

def rotate_to_global(imu_accel, imu_ang_vel, angle_deg):
    """
    将 IMU 坐标系的加速度和角速度转换到全局坐标系。

    Args:
        imu_accel: (N, 3) IMU 加速度数据
        imu_ang_vel: (N, 3) IMU 角速度数据
        angle_deg: 顺时针旋转角度（度）

    Returns:
        global_accel: (N, 3) 全局坐标系加速度
        global_ang_vel: (N, 3) 全局坐标系角速度
    """
    # 转换顺时针角度为逆时针角度（IMU 转到全局坐标系）
    angle_rad = np.radians(angle_deg)  # 顺时针为负角度

    # 绕 Z 轴的旋转矩阵
    rotation_matrix = R.from_euler('x', angle_rad).as_matrix()

    # 将 IMU 加速度和角速度旋转到全局坐标系
    global_accel = imu_accel @ rotation_matrix.T
    global_ang_vel = imu_ang_vel @ rotation_matrix.T

    return global_accel, global_ang_vel


def main():
    bag_path = '/media/crrcdt123/glam/slam_obstacle_detector_datasets/20250517/20250518-000903/'  # 替换为 bag 文件路径
    topic_name = '/C6/imu'  # 替换为 IMU 话题名称
    csv_file = '/media/crrcdt123/glam/slam_obstacle_detector_datasets/map/imu_data_20250518-000903.csv'
    # 读取 IMU 数据
    imu_df = read_imu_from_bag(bag_path, topic_name)
    # imu_df = pd.read_csv(csv_file)
    acc = np.column_stack((imu_df['Accel X (m/s²)'], imu_df['Accel Y (m/s²)'], imu_df['Accel Z (m/s²)']))
    ang = np.column_stack((imu_df['Gyro X (deg/s)'], imu_df['Gyro X (deg/s)'], imu_df['Gyro X (deg/s)']))
    # global_accel, global_ang_vel = rotate_to_global(acc, ang, 20)
    global_accel, global_ang_vel = acc, ang
    imu_df['Accel X (m/s²)'], imu_df['Accel Y (m/s²)'], imu_df['Accel Z (m/s²)'] = global_accel[:, 0], global_accel[:, 1], global_accel[:, 2]
    imu_df['Gyro X (deg/s)'], imu_df['Gyro Y (deg/s)'], imu_df['Gyro Z (deg/s)'] = global_ang_vel[:, 0], global_ang_vel[:, 1], global_ang_vel[:, 2]
    # imu_df['Gyro X (deg/s)'], imu_df['Gyro Y (deg/s)'], imu_df['Gyro Z (deg/s)'] = imu_df['Gyro X (deg/s)']/np.pi * 180, imu_df['Gyro Y (deg/s)']/np.pi * 180, imu_df['Gyro Z (deg/s)']/np.pi * 180
    # 打印表格
    print(imu_df)
    imu_df.to_csv(csv_file, index=False)
    # 可视化表格
    try:
        import matplotlib.pyplot as plt
        imu_df.plot(x='Timestamp (s)', y=['Accel X (m/s²)', 'Accel Y (m/s²)', 'Accel Z (m/s²)'], title='Linear Acceleration')
        imu_df.plot(x='Timestamp (s)', y=['Gyro X (deg/s)', 'Gyro Y (deg/s)', 'Gyro Z (deg/s)'], title='Angular Velocity')
        plt.show()
    except ImportError:
        print("Matplotlib is not installed. Skipping visualization.")

if __name__ == "__main__":
    main()
