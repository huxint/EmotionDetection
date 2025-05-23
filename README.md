# 实时人脸表情识别系统

这是一个使用PyTorch和OpenCV实现的实时人脸表情识别系统。该系统可以实时检测摄像头中的人脸，并识别其表情（如高兴、悲伤、惊讶等）。

## 功能特点

- 实时摄像头视频流捕获
- 人脸检测
- 表情识别（7种基本表情）
- 实时显示识别结果
- 人脸边界框标注

## 环境要求

- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- Pillow

## 安装步骤

1. 克隆项目到本地
2. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 确保已安装所有依赖包
2. 运行主程序：
```bash
python emotion_detection.py
```
3. 按'q'键退出程序

## 注意事项

- 确保摄像头可用
- 保持良好的光照条件以获得更好的识别效果 