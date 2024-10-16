import sys
# 将父目录添加到系统路径
sys.path.append('../cifar10_models/')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# 导入PreActResNet18
from preact_resnet import PreActResNet18

# 超参数设置
num_epochs = 200
batch_size = 128
learning_rate = 0.01

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=True, download=False, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=False, download=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# 定义设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 实例化模型
model = PreActResNet18().to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 训练模型
def train(epoch):
    print(f'\nEpoch: {epoch}')
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            # print(f'Batch {batch_idx}/{len(trainloader)} | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(trainloader)}], Loss: {loss.item():.4f}, Accuracy: {100. * correct / total:.2f}%')

# 测试模型
def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    print(f'Test Loss: {test_loss/len(testloader):.3f} | Test Acc: {acc:.3f}%')

    # 保存最佳模型
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, '../../checkpoints/cifar10_target_models/PreActResNet_ckpt.t7')
        best_acc = acc

best_acc = 0

if __name__ == '__main__':
    print('The train is start......\n')
    # Training loop
    for epoch in range(num_epochs):
        train(epoch)
        test(epoch)
        scheduler.step()


