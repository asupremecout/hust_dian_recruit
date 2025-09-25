import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# 残差块
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1,
                               stride) if stride != 1 or in_channels != out_channels else None

    # 输出尺寸 = (输入尺寸 - 内核大小 + 2 * 填充) / 步长 + 1
    # 2个conv2d函数只有一次用了stride，且stride为外界传入，所以如果传入一，则整个一块shape不变
    # stride如果不为1，用downsample传入stride，保证x.shape一致

    # 2个conv2d函数最终输出尺寸为out_channels，如果self.in_channels != out_channels * BasicBlock.expansion（最终channel）
    # 则要用downsample nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,kernel_size=1, stride=stride, bias=False)
    # 保证channel一致

    # 通过上述设计，保证y和x的形状一致，可以相加
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        x = self.conv3(x) if self.conv3 else x
        return F.relu(y + x)


# ResNet-18模型
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)  # kernelsize为7x7, stride=2，padding=3，
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)  # 最大池化
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    #1个conv1有一层卷积层，layer1234每个各有2*2个卷积层，再加上最后的全连接层，共18层
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [Residual(in_channels, out_channels, stride)]
        for _ in range(1, blocks): layers.append(Residual(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)  # 添加池化
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(self.avgpool(x).view(x.size(0), -1))


# 训练和测试
def main():
    # 数据准备
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    # 模型设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 检测是否有可用的GPU，如果有则使用GPU，否则使用CPU
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    train_losses, train_accs = [], []
    for epoch in range(10):
        model.train()
        loss_sum, correct, total = 0, 0, 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        train_losses.append(loss_sum / len(trainloader))
        train_accs.append(acc)
        print(f'Epoch {epoch + 1}: Loss: {train_losses[-1]:.3f}, Acc: {acc:.2f}%')

    # 测试
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    # 可视化
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_accs)
    plt.title('Training Accuracy')
    plt.show()


if __name__ == "__main__":
    main()