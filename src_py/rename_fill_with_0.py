import os

# 要处理的文件夹路径
root_dir = '/home/crrcdt123/datasets2/twoDoor/二门训练数据/datasets/train/'

# 遍历所有子文件夹
for name in os.listdir(root_dir):
    full_path = os.path.join(root_dir, name)
    if os.path.isdir(full_path):
        # 尝试将原名转换为数字
        try:
            new_name = str(int(name)).zfill(5)  # 转换为整数后补零
            new_path = os.path.join(root_dir, new_name)
            os.rename(full_path, new_path)
            print(f'Renamed {name} → {new_name}')
        except ValueError:
            print(f'Skip non-numeric folder: {name}')
