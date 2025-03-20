import os
import shutil

# 源目录
source_dir = '/data2/czh/datasets/Train/Deraindrop/NightRainDrop_Train/Clear/'

# 目标目录
target_dir = '/data2/czh/datasets/Train/CL_Deraindrop/gt/'

# 确保目标目录存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 初始化一个计数器，从 1 开始
counter = 4714

# 遍历源目录中的所有子文件夹
for subfolder in os.listdir(source_dir):
    subfolder_path = os.path.join(source_dir, subfolder)

    # 确保是文件夹
    if os.path.isdir(subfolder_path):
        # 遍历该子文件夹中的所有文件
        for file_name in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file_name)

            # 如果是图片文件（可以根据实际需要调整文件类型）
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # 生成新的文件名，按零填充到5位数，如 00001.jpg
                new_file_name = f"{counter:05d}{os.path.splitext(file_name)[-1]}"
                # 复制并重命名到目标文件夹
                shutil.copy(file_path, os.path.join(target_dir, new_file_name))
                # 更新计数器
                counter += 1

print("图片复制并重新编号完成！")
