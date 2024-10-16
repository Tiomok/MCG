import os
import sys
# 将父目录添加到系统路径
sys.path.append('../cifar10_models/')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from vgg import VGG  # 确保 vgg.py 文件在同一目录下或适当导入

# 定义超参数
batch_size = 32
num_epochs = 100
learning_rate = 0.001

# CIFAR-10 数据集的转换
transform = transforms.Compose([
    # transforms.Resize(224),  # VGG 需要输入224x224的图片
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练函数
def train(model, trainloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}')

# 测试函数
def test(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')

# main 函数
def main():
    # 下载和加载训练集和测试集
    trainset = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # 训练 VGG13
    vgg13_model = VGG('VGG13').to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vgg13_model.parameters(), lr=learning_rate)

    print("Training VGG13...")
    train(vgg13_model, trainloader, criterion, optimizer, num_epochs)
    torch.save(vgg13_model.state_dict(), 'VGG13_ckpt.t7')
    print("VGG13 model saved to VGG13_ckpt.t7")

    # 训练 VGG19
    vgg19_model = VGG('VGG19').to(device)
    optimizer = optim.Adam(vgg19_model.parameters(), lr=learning_rate)

    print("Training VGG19...")
    train(vgg19_model, trainloader, criterion, optimizer, num_epochs)
    torch.save(vgg19_model.state_dict(), 'VGG19_ckpt.t7')
    print("VGG19 model saved to VGG19_ckpt.t7")

    # 测试模型性能
    print("Testing VGG13...")
    test(vgg13_model, testloader)

    print("Testing VGG19...")
    test(vgg19_model, testloader)

# Python脚本入口
if __name__ == "__main__":
    main()
