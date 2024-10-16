import sys
# 将父目录添加到系统路径
sys.path.append('../cifar10_models/')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import os

# 加载DenseNet121模型
from densenet import DenseNet121  # 假设DenseNet模型保存在densenet.py中

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
num_epochs = 200
learning_rate = 0.01
batch_size = 64

# CIFAR-10数据集
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

train_dataset = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=True, download=False, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=False, download=False, transform=transform_test)

train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 模型
model = DenseNet121().to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 初始化最优准确率
best_acc = 0

# 尝试加载预训练的模型权重
checkpoint_path = '../../checkpoints/cifar10_target_models/DenseNet_ckpt.t7'
if os.path.exists(checkpoint_path):
    print(f'Loading checkpoint from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])  # 加载模型参数
    best_acc = checkpoint['acc']  # 加载最优的准确率
    start_epoch = checkpoint['epoch'] + 1  # 从保存的下一个epoch继续训练
else:
    print(f'No checkpoint found at {checkpoint_path}, starting from scratch...')
    start_epoch = 0

# 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')

# 测试函数
def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print(f'Test Accuracy: {acc:.2f}%')

    # 保存模型，如果当前准确率比之前的最好结果高
    if acc > best_acc:
        print(f'Saving model with accuracy {acc:.2f}%...')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/DenseNet_ckpt.t7')
        best_acc = acc


if __name__ == '__main__':
    print('The training process starts...\n')
    # 从start_epoch开始训练
    for epoch in range(start_epoch, num_epochs):
        train(epoch)
        test(epoch)
        scheduler.step()
