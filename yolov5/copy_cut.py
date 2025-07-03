import os
import cv2
import numpy as np

# 输入和输出目录
input_image_dir = "runs/detect/exp5"
label_dir = "/data/shentongtong/public_datasets/SatNet_Augment/labels/test"
output_crop_dir = "output_crops"

# 创建输出目录
os.makedirs(output_crop_dir, exist_ok=True)

# 遍历所有图片
for image_name in os.listdir(input_image_dir):
    if not image_name.endswith(('.jpg', '.png', '.jpeg')):
        continue

    # 读取图片
    image_path = os.path.join(input_image_dir, image_name)
    image = cv2.imread(image_path)
    if image is None:
        continue

    # 获取对应的标签文件（YOLO 格式：class x_center y_center width height）
    label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + ".txt")
    if not os.path.exists(label_path):
        continue

    # 读取标签并裁剪
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        # 解析 YOLO 格式的 bbox（归一化坐标）
        class_id, x_center, y_center, width, height = map(float, parts)

        # 转换为像素坐标
        img_h, img_w = image.shape[:2]
        x_center *= img_w
        y_center *= img_h
        width *= img_w
        height *= img_h

        width *= 1.2
        height *= 1.2

        # 计算 bbox 的左上角和右下角
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # 确保坐标在图片范围内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)

        # 裁剪并保存
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # 保存裁剪后的图片（格式：原图名_类别_序号.jpg）
        crop_name = f"{os.path.splitext(image_name)[0]}_class{int(class_id)}_{i}.jpg"
        cv2.imwrite(os.path.join(output_crop_dir, crop_name), crop)

print("裁剪完成！")