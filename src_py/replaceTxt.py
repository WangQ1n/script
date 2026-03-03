import os

# 替换规则
replace_dict = {
    "1": "0",
    "7": "6"
}

# 文件夹路径
folder_path = "/home/crrcdt123/git/script/output/labels/train/"  # 替换为你的真实路径

# 遍历文件夹中所有txt文件
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)

        # 读取原文件
        with open(file_path, "r") as f:
            lines = f.readlines()

        # 处理每一行
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts and parts[0] in replace_dict:
                parts[0] = replace_dict[parts[0]]
            new_lines.append(" ".join(parts))

        # 覆盖写入
        with open(file_path, "w") as f:
            f.write("\n".join(new_lines) + "\n")
