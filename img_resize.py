# -*- coding: utf-8 -*-
import cv2
import os

# opencv自带的一个面部识别分类器
detection_model_path = 'model/haarcascade_frontalface_default.xml'

# 加载人脸检测模型
face_detection = cv2.CascadeClassifier(detection_model_path)

# 设置源文件夹和目标文件夹
source_folder = 'face_images/resize_input'
target_folder = 'face_images/resize_output'

# 确保目标文件夹存在
os.makedirs(target_folder, exist_ok=True)


# 遍历源文件夹中的所有图片
for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        # 打开图片
        img_path = os.path.join(source_folder, filename)
        #  读取输入图像
        img = cv2.imread(img_path)
        # 获取当前帧中的全部人脸
        faces = face_detection.detectMultiScale(img, 1.3, 5)

        i = 0
        for (x, y, w, h) in faces:
            # 获取人脸图像
            face = img[y:y + h, x:x + w]
            try:
                # shape变为(48,48)
                face = cv2.resize(face, (48, 48))
            except:
                continue

            # 输出图像
            # 构建新的文件名
            new_file_name = f"{i + 1}、{filename}"
            output_path = os.path.join(target_folder, new_file_name)
            # cv2.imwrite(output_path, face)
            cv2.imencode('.jpg', face)[1].tofile(output_path)
            i = i + 1

print("图片处理完成！")
