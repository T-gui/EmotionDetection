import os
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda")


class FacialDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.images = self.load_images()

    def load_images(self):
        images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                images.append((image_path, self.class_to_idx[class_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label


data_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

train_dir = 'dataset/train'
test_dir = 'dataset/val'
train_dataset = FacialDataset(root_dir=train_dir, transform=data_transforms)
test_dataset = FacialDataset(root_dir=test_dir, transform=data_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义一个卷积神经网络
class CNNModel(nn.Module):
    def __init__(self, numb_classes=7, dropout_prob=0.3, leaky_relu_slope=0.01):
        super(CNNModel, self).__init__()
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, numb_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.conv_block1(x)))
        x = self.pool(self.leaky_relu(self.conv_block2(x)))
        x = self.pool(self.leaky_relu(self.conv_block3(x)))
        x = self.pool(self.leaky_relu(self.conv_block4(x)))
        x = x.reshape(-1, 512 * 6 * 6)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 训练模型
def train(num_epochs, train_dataloader, test_dataloader):
    model = CNNModel()
    model = model.to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.015)  # 优化器为Adam

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # 设置为训练模式
                dataloader = train_dataloader
            else:
                model.eval()  # 设置为评估模式
                dataloader = test_dataloader

            running_loss = 0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播 + 优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                epoch_loss = running_loss / len(train_dataset)
                epoch_acc = running_corrects / len(train_dataset)
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.cpu())
            else:
                epoch_loss = running_loss / len(test_dataset)
                epoch_acc = running_corrects / len(test_dataset)
                test_losses.append(epoch_loss)
                test_accs.append(epoch_acc.cpu())
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # 绘制损失曲线
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.show()

    # 绘制准确率曲线
    plt.figure()
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.show()
    print('Training complete!')
    torch.save(model, 'model.pkl')
    serialized_model = torch.jit.script(model)
    serialized_model.save('model.pt')


if __name__ == '__main__':
    train(num_epochs=60, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
