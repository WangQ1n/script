import os
import pandas as pd

def rename_files_based_on_excel(folder_path, excel_file_path, sheet_name, old_name_column, new_name_column):
    # 读取要用作映射的 Excel 文件
    excel_data = pd.read_excel(excel_file_path, sheet_name=sheet_name)

    # 遍历文件夹中的所有 Excel 文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            file_path = os.path.join(folder_path, filename)
            name = int(filename.split(".")[0])
            # 读取当前文件的 Excel 数据
            # file_data = pd.read_excel(file_path, sheet_name=sheet_name)

            # 获取对应的新文件名
            # matching_row = excel_data[excel_data[old_name_column] == file_data.iloc[0][old_name_column]]
            try:
                row_index = excel_data.index[excel_data["Unnamed: 2"] == name].tolist()
                print(f"目标数据 {name} 位于第 {row_index} 行")
            except IndexError:
                print(f"未找到目标数据 {name}")
            if len(row_index) == 1:
                row_index = row_index[0]
                order = excel_data.iloc[row_index, 5]
                if order == "高华鑫":
                    new_name = excel_data.iloc[row_index, 3] + "数据表"
                    print(new_name)
                    #构建新的文件路径
                    new_file_path = os.path.join(folder_path, new_name + ".xlsx")
                    #重命名文件
                    os.rename(file_path, new_file_path)
                    print(f"文件重命名: {filename} -> {new_name}.xlsx")
            elif len(row_index) > 1:
                print(f"目标数据 {name} 位于第 {row_index} 行")


# 示例用法
folder_path = "/home/crrcdt123/Downloads/20240124设备描述_数据表/1PE压力管道"
excel_file_path = "/home/crrcdt123/Downloads/20240124设备描述_数据表/扬子石化B级设备自查情况20240124-设备分工.xlsx"
sheet_name = "Sheet1 (2)"
old_name_column = 3
new_name_column = 4

rename_files_based_on_excel(folder_path, excel_file_path, sheet_name, old_name_column, new_name_column)