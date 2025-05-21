import os
import requests
import zipfile
import pandas as pd
import numpy as np
from PIL import Image
import shutil
import urllib.request
import cv2
import kaggle

def download_fer2013():
    print("开始下载FER2013数据集...")
    
    # 创建临时目录
    if not os.path.exists('temp'):
        os.makedirs('temp')
    
    # 下载数据集
    # print("正在下载数据集...")
    # try:
    #     # 使用kaggle API下载数据集
    #     kaggle.api.dataset_download_files('msambare/fer2013', path='temp', unzip=True)
    # except Exception as e:
    #     print("自动下载失败，请按照以下步骤手动下载：")
    #     print("1. 访问 https://www.kaggle.com/datasets/msambare/fer2013")
    #     print("2. 点击 'Download' 按钮下载数据集")
    #     print("3. 解压下载的文件，将 'train' 和 'test' 文件夹放在 'temp' 目录下")
    #     input("完成上述步骤后按回车键继续...")
    
    # 创建表情数据集目录
    if not os.path.exists('emotion_dataset'):
        os.makedirs('emotion_dataset')
    
    # 英文类别到中文类别的映射
    emotion_map = {
        'angry': '生气',
        'disgust': '厌恶',
        'fear': '恐惧',
        'happy': '高兴',
        'neutral': '平静',
        'sad': '悲伤',
        'surprise': '惊讶'
    }
    
    # 创建表情类别目录
    for emotion in emotion_map.values():
        emotion_dir = os.path.join('emotion_dataset', emotion)
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)
    
    # 处理训练集
    print("正在处理训练集...")
    train_dir = 'temp/train'
    for eng, chi in emotion_map.items():
        emotion_dir = os.path.join(train_dir, eng)
        if os.path.exists(emotion_dir):
            for img_name in os.listdir(emotion_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    src_path = os.path.join(emotion_dir, img_name)
                    dst_path = os.path.join('emotion_dataset', chi, f'train_{img_name}')
                    shutil.copy2(src_path, dst_path)
    
    # 处理测试集
    print("正在处理测试集...")
    test_dir = 'temp/test'
    for eng, chi in emotion_map.items():
        emotion_dir = os.path.join(test_dir, eng)
        if os.path.exists(emotion_dir):
            for img_name in os.listdir(emotion_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    src_path = os.path.join(emotion_dir, img_name)
                    dst_path = os.path.join('emotion_dataset', chi, f'test_{img_name}')
                    shutil.copy2(src_path, dst_path)
    
    # 清理临时文件
    print("清理临时文件...")
    shutil.rmtree('temp')
    
    print("数据集处理完成！")
    print("数据集已保存在 emotion_dataset 目录下")
    print("注意：FER2013数据集包含约35,000张人脸表情图片，分为训练集和测试集。")

if __name__ == '__main__':
    download_fer2013()