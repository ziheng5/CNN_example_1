# 导入PyTorch库
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 定义LeNet-5架构的神经网络类
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 前向传播函数定义网络的数据流向
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)      # 展平为一维
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = self.fc3(x)
        x = torch.nn.functional.log_softmax(self.fc3(x), dim=1)
        return x


print('开始加载数据集')
# 定义数据变换和加载MNIST数据集，数据预处理
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])   # 标准化处理，均值0.5，方差0.5

# 训练数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 测试数据集
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print('加载完成')

# 初始化LeNet-5模型以及定义损失函数和优化器
net = LeNet5()#创建实例
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，用于分类问题
optimizer = optim.Adam(net.parameters(), lr=0.001)  # Adam，学习率为0.001，
# optimizer = optim.SGD(net.parameters(), lr=0.001)   # SGD，学习率为0.001

# 训练循环
for epoch in range(10):  # 可以根据需要调整训练的轮数
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()  # 清零梯度

        outputs = net.forward(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新权重

        running_loss += loss.item()

    print(f"Epoch {epoch + 1},  Loss: {running_loss / len(train_loader)}")

print("Finished Training")

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = net(inputs)  # 前向传播
        _, predicted = torch.max(outputs.data, 1)  # 找到最大概率的类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on the test set: {accuracy}%")