import os
import shutil

# 原始数据集路径
gt_dir = "/data2/czh/datasets/Test/CL_Deraindrop/gt"
drop_dir = "/data2/czh/datasets/Test/CL_Deraindrop/drop"

# 目标数据集路径
target_dirs = [
    "/data2/czh/datasets/Test/CL_Deraindrop1",
    "/data2/czh/datasets/Test/CL_Deraindrop2",
    "/data2/czh/datasets/Test/CL_Deraindrop3",
]

# 获取所有图片文件，并排序保证一致
gt_files = sorted(os.listdir(gt_dir))
drop_files = sorted(os.listdir(drop_dir))

# 确保文件名一一对应
assert len(gt_files) == len(drop_files), "GT and drop images count don't match!"

# 计算每份的大小
total_files = len(gt_files)
split_size = total_files // 3

# 创建目标文件夹
for target_dir in target_dirs:
    os.makedirs(os.path.join(target_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "drop"), exist_ok=True)

# 按顺序分配数据
for i, (gt_file, drop_file) in enumerate(zip(gt_files, drop_files)):
    subset_idx = i // split_size  # 计算当前样本应该放在哪个子集
    subset_idx = min(subset_idx, 2)  # 确保所有数据都分配到三个子集之一

    target_gt_path = os.path.join(target_dirs[subset_idx], "gt", gt_file)
    target_drop_path = os.path.join(target_dirs[subset_idx], "drop", drop_file)

    shutil.move(os.path.join(gt_dir, gt_file), target_gt_path)
    shutil.move(os.path.join(drop_dir, drop_file), target_drop_path)

print("数据集已成功划分为三份！")
