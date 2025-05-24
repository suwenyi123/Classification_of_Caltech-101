import os
import shutil
import random
from tqdm import tqdm

data_root = "/home/wenyisu/NNandDL/second/1/dataset/images/101_ObjectCategories"
train_dir = "/home/wenyisu/NNandDL/second/1/dataset/images/train"
test_dir = "/home/wenyisu/NNandDL/second/1/dataset/images/test"

# 创建目标目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 遍历每个类别
for class_name in tqdm(os.listdir(data_root)):
    class_path = os.path.join(data_root, class_name)
    if not os.path.isdir(class_path):
        continue

    # 获取所有图片路径
    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.seed(42)  # 固定随机种子确保可复现
    random.shuffle(images)

    # 计算分割点（80%训练，20%测试）
    split_idx = int(0.8 * len(images))
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # 创建类别子目录
    train_class_path = os.path.join(train_dir, class_name)
    test_class_path = os.path.join(test_dir, class_name)
    os.makedirs(train_class_path, exist_ok=True)
    os.makedirs(test_class_path, exist_ok=True)

    # 移动文件
    for img in train_images:
        shutil.move(os.path.join(class_path, img), train_class_path)
    for img in test_images:
        shutil.move(os.path.join(class_path, img), test_class_path)

print("数据集拆分完成！")