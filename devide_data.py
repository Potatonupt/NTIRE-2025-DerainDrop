import os
import random
import shutil

# 定义文件夹路径
gt_dir = '/data2/czh/datasets/Train/CL_Deraindrop/gt'
drop_dir = '/data2/czh/datasets/Train/CL_Deraindrop/drop'

# 测试集存储路径
test_gt_dir = '/data2/czh/datasets/Test/CL_Deraindrop/gt'
test_drop_dir = '/data2/czh/datasets/Test/CL_Deraindrop/drop'

# 获取所有图片文件
gt_files = os.listdir(gt_dir)
drop_files = os.listdir(drop_dir)

# 确保文件名一一对应
assert len(gt_files) == len(drop_files), "GT and drop images count don't match!"

# 随机选择2000张图片作为测试集
test_indices = random.sample(range(len(gt_files)), 2000)

# 创建测试集目录
os.makedirs(test_gt_dir, exist_ok=True)
os.makedirs(test_drop_dir, exist_ok=True)

# 将选中的图片复制到测试集目录，并删除原图
for index in test_indices:
    gt_file = gt_files[index]
    drop_file = drop_files[index]

    # 复制文件到测试集目录
    shutil.copy(os.path.join(gt_dir, gt_file), os.path.join(test_gt_dir, gt_file))
    shutil.copy(os.path.join(drop_dir, drop_file), os.path.join(test_drop_dir, drop_file))

    # 删除原来的图片
    os.remove(os.path.join(gt_dir, gt_file))
    os.remove(os.path.join(drop_dir, drop_file))

print("测试集已成功划分并删除原图片！")
