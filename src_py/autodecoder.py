import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import re
import os


def parse_point(input_str):
    # 去掉方括号
    match = re.match(r'-?\s*\[\s*([0-9.-]+)\s*,\s*([0-9.-]+)\s*\]', input_str)
    
    if not match:
        return None

    # 获取 x 和 y
    x_str, y_str = match.groups()

    # 将 x 和 y 转换为浮点数并进行四舍五入
    x = round(float(x_str))
    y = round(float(y_str))

    # 返回一个元组 (x, y)
    return (x, y)


# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None):
        self.transform = transform
        self.image_paths = self.load_image_paths(image_path)
        self.labels = self.load_labels(label_path)
        

    def load_image_paths(self, path):
        from pathlib import Path
        # 文件夹路径
        folder_path = Path(path)
        # 获取所有 .jpg 文件的路径
        jpg_files = list(folder_path.glob('*.jpg'))
        return jpg_files

    def load_labels(self, path):
        labels = {}
        with open(path, 'r') as file:
            lines = file.readlines()
        cam_name = ""
        polygon = []
        for line in lines:
            line = line.strip()
            if '.jpg' in line:
                # Extract camera name
                pos = line.find('-')
                cam_name = line[:pos]
                is_new_cam = True
            elif 'roi' in line or line == "":
                if polygon:
                    if cam_name not in labels:
                        labels[cam_name] = []
                    labels[cam_name].append(polygon)
                    polygon = []
            else:
                pt = parse_point(line)
                if pt:
                    polygon.append(pt)
        return labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        cam_name = img_path.name.split("-")[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)

        # 创建一个空的黑色图像作为掩码
        mask = np.zeros(img.shape, dtype=np.uint8)

        contour = np.array(self.labels[cam_name][0])
        # 绘制多边形轮廓在掩码上（使用白色填充）
        cv2.fillPoly(mask, [contour], (255, 255, 255))

        # 提取图像中轮廓区域的内容
        img = cv2.bitwise_and(img, mask)
        x, y, w, h = cv2.boundingRect(contour)

        # 使用外接矩形的坐标裁切图像
        img = img[y:y+h, x:x+w]
        img = cv2.resize(img, (128, 128))
        # img = np.expand_dims(img, axis=-1)  # 加入颜色通道
        # cv2.imshow("roi0", img)
        # cv2.waitKey(0)
        img = img.astype('float32') / 255.0  # 归一化到[0, 1]
        if self.transform:
            img = self.transform(img)

        return img

# 自编码器模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 输出尺寸: 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 输出尺寸: 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 输出尺寸: 16x16
            nn.ReLU()
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出尺寸: 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出尺寸: 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出尺寸: 128x128
            nn.Sigmoid()  # 输出值在[0, 1]之间
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 保存原图和重构图的效果图
def save_reconstruction_images(original, reconstructed, epoch, save_path):
    original = original.cpu().detach().numpy()[0, ...]  # 从 Tensor 转为 NumPy 数组
    reconstructed = reconstructed.cpu().detach().numpy()[0, ...]

    # 转为 [0, 255] 范围并转换为 uint8 类型
    original = np.clip(original * 255, 0, 255).astype(np.uint8)
    reconstructed = np.clip(reconstructed * 255, 0, 255).astype(np.uint8)
    original = np.transpose(original, (1, 2, 0))
    reconstructed = np.transpose(reconstructed, (1, 2, 0))
    # 创建并保存效果图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(original)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(reconstructed)
    ax2.set_title('Reconstructed Image')
    ax2.axis('off')

    plt.savefig(os.path.join(save_path, f'epoch_{epoch+1}.png'))
    plt.close()

# 计算重构误差
def compute_reconstruction_error(original, reconstructed):
    # 计算原图像和重构图像的均方误差
    mse = torch.mean((original - reconstructed) ** 2)
    return mse

# 训练自编码器
def train_autoencoder(model, dataloader, num_epochs=10, learning_rate=1e-3):
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in dataloader:
            data = data.to(device)

            optimizer.zero_grad()

            # 前向传播
            reconstructed = model(data)

            # 计算损失
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 1 == 0:  # 每个 epoch 保存一次
            save_reconstruction_images(data, reconstructed, epoch, "/home/crrcdt123/二门数据/reconstructed/")
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')

# 在测试图像上检测瑕疵
def detect_anomalies(model, image_path, labels, threshold=0.02):
    cam_name = image_path.split("-")[0]
    cam_name = cam_name.split("/")[-1]
    img = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
    # 创建一个空的黑色图像作为掩码
    mask = np.zeros(img.shape, dtype=np.uint8)

    contour = np.array(labels[cam_name][0])
    # 绘制多边形轮廓在掩码上（使用白色填充）
    cv2.fillPoly(mask, [contour], (255, 255, 255))

    # 提取图像中轮廓区域的内容
    img = cv2.bitwise_and(img, mask)
    x, y, w, h = cv2.boundingRect(contour)

    # 使用外接矩形的坐标裁切图像
    img = img[y:y+h, x:x+w]
    img = cv2.resize(img, (128, 128))
    # cv2.imshow("test", img)
    # cv2.waitKey(0)
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32') / 255.0  # 归一化到[0, 1]
    img_tensor = torch.tensor(img).unsqueeze(0).to("cuda")  # 转为 tensor 并添加 batch 维度

    model.eval()
    with torch.no_grad():
        reconstructed = model(img_tensor)
    
    # 计算重构误差
    reconstruction_error = torch.mean((img_tensor - reconstructed) ** 2)
    print(f'Reconstruction error: {reconstruction_error.item()}')

    # 将重构误差大于阈值的区域视为瑕疵
    if reconstruction_error.item() > threshold:
        print("瑕疵检测：图像有瑕疵")
    else:
        print("瑕疵检测：图像正常")
    original = img_tensor.cpu().detach().numpy()[0, ...]  # 从 Tensor 转为 NumPy 数组
    reconstructed = reconstructed.cpu().detach().numpy()[0, ...]

    # 转为 [0, 255] 范围并转换为 uint8 类型
    original = np.clip(original * 255, 0, 255).astype(np.uint8)
    reconstructed = np.clip(reconstructed * 255, 0, 255).astype(np.uint8)
    original = np.transpose(original, (1, 2, 0))
    reconstructed = np.transpose(reconstructed, (1, 2, 0))
    diff = cv2.absdiff(original, reconstructed)
    # 创建并保存效果图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    ax1.imshow(original)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(reconstructed)
    ax2.set_title('Reconstructed Image')
    ax2.axis('off')

    ax3.imshow(diff)
    ax3.set_title('Diff Image')
    ax3.axis('off')
    plt.show()


# 主程序
if __name__ == '__main__':
    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载训练数据集
    images_path = "/home/crrcdt123/二门数据/1012_delay_1.5"  # 图像路径
    label_path = "/home/crrcdt123/二门数据/images_labels/infos.txt"
    transform = transforms.ToTensor()
    dataset = ImageDataset(images_path, label_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 初始化并训练自编码器
    model = Autoencoder().to(device)
    # train_autoencoder(model, dataloader)

    # 保存训练好的模型
    # torch.save(model.state_dict(), 'autoencoder.pth')

    model.load_state_dict(torch.load('autoencoder.pth'))
    model.eval()
    # 在新图像上检测瑕疵
    detect_anomalies(model, '/home/crrcdt123/二门数据/cam113-End-20241024-194022579.jpg', dataset.labels, threshold=0.02)
