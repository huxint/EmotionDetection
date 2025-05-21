import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from PIL import ImageFont, ImageDraw

# 定义表情类别
EMOTIONS = ['生气', '厌恶', '恐惧', '高兴', '悲伤', '惊讶', '平静']

def draw_chinese_text(img, text, position, color=(0,255,0), font_size=24):
    # OpenCV的BGR转RGB
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # 你可以换成自己系统支持的中文字体路径
    font = ImageFont.truetype("simhei.ttf", font_size)
    draw.text(position, text, font=font, fill=color)
    # RGB转回BGR
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 定义改进的CNN模型
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, len(EMOTIONS))
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.fc(x)
        return x

def load_model(model_path):
    model = EmotionCNN()
    try:
        model.load_state_dict(torch.load(model_path))
        print("模型加载成功！")
    except:
        print("模型加载失败，使用随机初始化的模型")
    model.eval()
    return model

def preprocess_face(face_img):
    # 转换为灰度图
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    # 调整大小为48x48
    resized = cv2.resize(gray, (48, 48))
    # 转换为PIL图像
    pil_img = Image.fromarray(resized)
    # 转换为张量
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    tensor = transform(pil_img)
    return tensor.unsqueeze(0)

def main():
    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 加载表情识别模型
    model = load_model('emotion_model.pth')

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # 提取人脸区域
            face_roi = frame[y:y+h, x:x+w]

            # 预处理人脸图像
            face_tensor = preprocess_face(face_roi)

            # 进行表情预测
            with torch.no_grad():
                outputs = model(face_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                emotion = EMOTIONS[predicted.item()]
                conf = confidence.item()

            # 绘制边界框和标签
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{emotion} ({conf:.2f})"
            # cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            frame = draw_chinese_text(frame, label, (x, y-30), color=(0,255,0), font_size=24)

        # 显示结果
        cv2.imshow('Emotion Detection', frame)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()