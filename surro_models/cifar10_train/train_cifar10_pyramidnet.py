import sys
# 将父目录添加到系统路径
sys.path.append('../cifar10_models/')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pyramidnet import pyramid_net110

# 超参数
batch_size = 256
learning_rate = 0.1
num_epochs = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

train_dataset = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=True, download=False, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=False, download=False, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

# 模型
model = pyramid_net110().to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


# 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

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
            print(
                f'Epoch [{epoch}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {running_loss / (batch_idx + 1):.4f}, Accuracy: {100. * correct / total:.2f}%')


# 测试函数
def test():
    model.eval()
    test_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f'Test Loss: {test_loss / (batch_idx + 1):.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy


# 主训练循环
def main():
    best_acc = 0  # 记录最佳精度
    for epoch in range(1, num_epochs + 1):
        train(epoch)
        accuracy = test()
        scheduler.step()

        # 每10个epoch保存一次模型
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'./pyramidnet_cifar10_epoch_{epoch}.pth')

        # 保存最佳模型
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
            }, './PyramidNet_ckpt.t7')

    # 训练结束后保存最终模型
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': best_acc,
    }, '../../checkpoints/cifar10_target_models/PyramidNet_ckpt.t7')


if __name__ == '__main__':
    main()

