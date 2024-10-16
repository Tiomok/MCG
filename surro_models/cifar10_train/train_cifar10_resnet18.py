import os
import sys
# 将父目录添加到系统路径
sys.path.append('../cifar10_models/')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from resnet import ResNet18

print("begin\n")

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载数据集
trainset = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
print("cifar10 Verification successful\n")

# 定义模型、损失函数和优化器
net = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# 学习率调整
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 初始化保存最好模型的相关变量
best_acc = 0.0  # 用于保存测试集上最高准确率
best_model_path = '../../checkpoints/temp/ResNet18_ckpt.t7'

'''
# 加载之前保存的模型和优化器状态
start_epoch = 0  # 初始化起始epoch
if os.path.exists(best_model_path):
    print(f'Loading checkpoint from {best_model_path}')
    checkpoint = torch.load(best_model_path)
    net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optim'])
    start_epoch = checkpoint['epoch'] + 1  # 从上次保存的epoch继续训练
    print(f'Resumed from epoch {start_epoch}')
'''

# 训练模型
def train(epoch):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:  # 每100个batch打印一次信息
            print(f'Batch {batch_idx}, Loss: {train_loss/(batch_idx+1):.3f}, Acc: {100.*correct/total:.3f}%')

# 测试模型
def test(epoch):
    global best_acc  # 使用全局变量保存最好准确率
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print(f'Test Loss: {test_loss/(batch_idx+1):.3f}, Test Acc: {acc:.3f}%')

    # 如果当前epoch的测试集准确率比之前的最好准确率高，保存模型
    if acc > best_acc:
        print(f'Saving Best Model with Test Acc: {acc:.3f}%')
        best_acc = acc
        torch.save({
            'model': net.state_dict(),
            'optim': optimizer.state_dict(),
            'epoch': epoch
        }, best_model_path)

if __name__ == "__main__":
    print("Already train ResNet18 on CIFAR10\n")
    # 训练多个epoch
    for start_epoch in range(200):
        train(start_epoch)
        test(start_epoch)
        scheduler.step()

