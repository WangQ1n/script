import os
import re
import time
import psutil
import logging
import platform
import subprocess
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler

# 设置日志配置， 30天一份日志，保留最近12分日志，午夜切割
log_dir = ""
log_path = os.path.join(log_dir, "memory_monitor.log")
# 设置midnight、interval=30 无法30天切割文件，故设置D
handler = TimedRotatingFileHandler(log_path, when='D', interval=30, backupCount=12, encoding="utf-8")
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)
# platform
architecture = platform.machine()
# 间隔时间
delay = 60  # s
# 关闭时间限制
# 设置起始时间为 2025-02-20
start_date = datetime(2025, 2, 20)
# 定义最大运行时间为365天
max_duration = timedelta(days=365)
# 设置阈值
MEMORY_THRESHOLD = 3.5 * 1024 * 1024 * 1024  # 内存阈值GB
# 要监控的进程名
TARGET_PROCESSES = ['railway_segment_node', 'objinfer', 'video_publisher']

# 打印信息
print("platform:%s, process:[%s, %s], railway uplimit:%.2fMB, log path:%s" % (architecture, TARGET_PROCESSES[0], TARGET_PROCESSES[1], round(MEMORY_THRESHOLD/1024/1024, 2), log_path))


# 获取进程占用内存信息
def get_process_memory_info(pid):
    try:
        p = psutil.Process(pid)
        memory_info = p.memory_info()
        return memory_info.rss, memory_info.vms, memory_info.shared
    except psutil.NoSuchProcess:
        return None, None, None

def get_process_swap_usage(pid):
    try:
        # 打开进程的 /proc/[pid]/status 文件
        with open(f'/proc/{pid}/status', 'r') as f:
            for line in f:
                if line.startswith('VmSwap'):
                    # 获取交换空间的大小
                    swap_usage = line.split()[1]  # swap usage is the second item in the line
                    return int(swap_usage)*1024  # 交换空间是以 B 为单位
        print(f"VmSwap information not found for PID {pid}.")
        return 0
    except FileNotFoundError:
        print(f"Process with PID {pid} does not exist.")
        return 0
    except PermissionError:
        print(f"Permission denied to access process {pid}.")
        return 0
    
# 获取显存占用情况
def get_gpu_memory(pid):
    try:
        result = subprocess.check_output(f"nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits",
            shell=True).decode('utf-8')
        # 使用正则表达式提取 pid 和显存数据
        pattern = re.compile(r'(\d+),\s*(\d+)')
        for line in result.splitlines():
            match = pattern.match(line.strip())
            if match and int(match.group(1)) == pid:
                used_memory = match.group(2)  # 获取显存数据
                return int(used_memory)*1024*1024  # 返回显存占用（单位：B）
        return 0  # 如果未找到指定进程
    except subprocess.CalledProcessError:
        return 0

# 监控内存并写入日志
def monitor_processes():
    is_info = False
    for proc in psutil.process_iter(attrs=['pid', 'name']):
        if proc.info['name'] in TARGET_PROCESSES:
            name = proc.info['name']
            pid = proc.info['pid']
            rss, vms, shared = get_process_memory_info(pid)
            swap = get_process_swap_usage(pid)
            if 'x86' in architecture:
                gpu_memory = get_gpu_memory(pid) 
            else:
                gpu_memory = 0  # jetson

            logger.info(f"{pid:<10} {round(vms/1024/1024, 2):<10} {round(rss/1024/1024, 2):<10} {round(swap/1024/1024, 2):<10} {round(gpu_memory/1024/1024, 2):<10} {name}")
            is_info = True
            # 检查是否超过阈值并采取行动
            total_memory = rss + swap + gpu_memory
            if name == "railway_segment_node" and total_memory > MEMORY_THRESHOLD:
                logger.error(f"Memory usage exceeded threshold for PID {pid}, NAME {name}. Killing process.")
                os.kill(pid, 9)  # 杀死进程,注意权限

    # 分隔符
    if is_info:
        logger.info("---------------------------------------------------------------------------")

# 定期监控进程（时间限制）
# def periodic_monitoring():
#     current_time = datetime.now()
#     while current_time - start_date < max_duration:
#         monitor_processes()
#         time.sleep(delay)  # 每n秒监控一次
#         current_time = datetime.now()
#     logger.warn("程序已超出最大运行时间（365天），将停止运行。")


# 定期监控进程
def periodic_monitoring():
    while True:
        monitor_processes()
        time.sleep(delay)  # 每n秒监控一次


if __name__ == "__main__":
    print("Starting process monitoring...")
    logger.info("Starting process monitoring...")
    header = f"PID        VM(MB)     RSS        Swap       GPU_Mem    Name"
    logger.info(header)
    periodic_monitoring()
