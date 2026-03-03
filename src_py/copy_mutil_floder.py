import os
import shutil
from pathlib import Path

def process_multiple_folders(folder_a, folder_b, folder_c, folder_d, folder_e, folder_f):
    """
    处理多个文件夹：从A获取文件名，从B提取到C，从D提取到E，在F创建空txt文件
    
    Args:
        folder_a: 源文件夹A（获取文件名）
        folder_b: 源文件夹B（提取文件到C）
        folder_c: 目标文件夹C（存放从B提取的文件）
        folder_d: 源文件夹D（提取文件到E）
        folder_e: 目标文件夹E（存放从D提取的文件）
        folder_f: 目标文件夹F（存放空txt文件）
    """
    # 创建所有目标文件夹（如果不存在）
    # for folder in [folder_c, folder_e, folder_f]:
    #     os.makedirs(folder, exist_ok=True)
    
    # 获取文件夹A中的所有文件名（不含扩展名）
    a_files = set()
    try:
        for file in os.listdir(folder_a):
            file_path = os.path.join(folder_a, file)
            if os.path.isfile(file_path):
                filename_without_ext = os.path.splitext(file)[0]
                a_files.add(filename_without_ext)
    except FileNotFoundError:
        print(f"错误：文件夹A不存在: {folder_a}")
        return
    except PermissionError:
        print(f"错误：没有权限访问文件夹A: {folder_a}")
        return
    
    print(f"文件夹A中找到 {len(a_files)} 个唯一文件名")
    
    # 处理函数：从源文件夹提取文件到目标文件夹
    def extract_files(source_folder, target_folder, folder_name):
        extracted_count = 0
        if not os.path.exists(source_folder):
            print(f"警告：{folder_name}文件夹不存在: {source_folder}")
            return extracted_count
        
        try:
            for file in os.listdir(source_folder):
                file_path = os.path.join(source_folder, file)
                if os.path.isfile(file_path):
                    filename_without_ext = os.path.splitext(file)[0]
                    
                    if filename_without_ext in a_files:
                        target_file = os.path.join(target_folder, "fake_" + file)
                        
                        # 复制文件
                        shutil.copy2(file_path, target_file)
                        extracted_count += 1
                        print(f"从{folder_name}提取: {file} -> {target_folder}")
            
            return extracted_count
        except Exception as e:
            print(f"处理{folder_name}时出错: {e}")
            return extracted_count
    
    # 从B文件夹提取文件到C文件夹
    b_to_c_count = extract_files(folder_b, folder_c, "B")
    
    # 从D文件夹提取文件到E文件夹
    d_to_e_count = extract_files(folder_d, folder_e, "D")
    
    # 在F文件夹创建空txt文件
    txt_count = 0
    try:
        for filename in a_files:
            filename = "fake_" + filename
            txt_file = os.path.join(folder_f, f"{filename}.txt")
            
            # 只在文件不存在时创建
            if not os.path.exists(txt_file):
                with open(txt_file, 'w', encoding='utf-8') as f:
                    pass  # 创建空文件
                txt_count += 1
                print(f"创建空txt: {filename}.txt -> {folder_f}")
    except Exception as e:
        print(f"创建txt文件时出错: {e}")
    
    # 输出统计信息
    print("\n" + "="*50)
    print("处理完成统计:")
    print(f"从B->C提取文件: {b_to_c_count} 个")
    print(f"从D->E提取文件: {d_to_e_count} 个")
    print(f"在F创建空txt文件: {txt_count} 个")
    print(f"A文件夹中总文件名: {len(a_files)} 个")

# 使用示例
if __name__ == "__main__":
    # 定义文件夹路径
    folder_a = "/home/crrcdt123/git/ultralytics/runs/segment/test34/"  # 参考文件名
    folder_b = "/media/crrcdt123/glam/crrc/data/su8/2door/083101-20250728/yolo/test"  # 源文件1
    folder_c = "/media/crrcdt123/glam/crrc/data/su8/2door/083101-20250728/dataset/images/train"  # 目标1（从B提取）
    folder_d = "/media/crrcdt123/glam/crrc/data/su8/2door/083101-20250728/yolo/test2"  # 源文件2
    folder_e = "/media/crrcdt123/glam/crrc/data/su8/2door/083101-20250728/dataset/images/train2"  # 目标2（从D提取）
    folder_f = "/media/crrcdt123/glam/crrc/data/su8/2door/083101-20250728/dataset/labels/train"  # 空txt文件
    
    # 执行处理
    process_multiple_folders(folder_a, folder_b, folder_c, folder_d, folder_e, folder_f)