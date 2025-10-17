import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO
from torchvision.models.resnet import ResNet, BasicBlock


# 定义 ResNet-10
class ResNet10(ResNet):
    def __init__(self, num_classes, in_channels=3, dropout_p=0.5):
        super().__init__(block=BasicBlock, layers=[1, 1, 1, 1])
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # 添加 Dropout 层（用于全连接层前）
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # 原有的前向传播逻辑
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # 在分类器前添加Dropout
        x = self.dropout1(x)
        x = self.fc(x)

        return x


def get_dataloaders(data_flag, batch_size=128, download=True):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    n_channels = info['n_channels']

    # 数据预处理
    if n_channels == 1:
        # 灰度图 → 复制成 3 通道
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        in_channels = 3
    else:
        # 已经是 RGB
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5] * 3, std=[.5] * 3)
        ])
        in_channels = 3

    # 数据集
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    val_dataset = DataClass(split='val', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader, len(info['label']), in_channels


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.squeeze().long().to(device)  # 保证是 (N,)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, criterion, device, schedular):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    schedular.step()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def fuse_conv_bn(conv, bn):
    W = conv.weight
    if conv.bias is None:
        b = torch.zeros(W.size(0), device=W.device)
    else:
        b = conv.bias
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    W_fused = W * (gamma / torch.sqrt(var + eps)).reshape([-1, 1, 1, 1])
    b_fused = beta + (b - mean) * gamma / torch.sqrt(var + eps)
    fused_conv = nn.Conv2d(conv.in_channels, conv.out_channels,
                           kernel_size=conv.kernel_size,
                           stride=conv.stride,
                           padding=conv.padding,
                           bias=True)
    fused_conv.weight.data.copy_(W_fused)
    fused_conv.bias.data.copy_(b_fused)
    return fused_conv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_flag', type=str, default='organamnist',
                        help='MedMNIST subset, e.g. pathmnist, bloodmnist, organamnist_axial')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # 数据集
    train_loader, val_loader, test_loader, num_classes, in_channels = get_dataloaders(
        args.data_flag, args.batch_size
    )

    # 模型
    model = ResNet10(num_classes=num_classes, in_channels=in_channels).to(args.device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)  # 增强正则化
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #
    # # 训练
    # best_val_acc = 0
    # for epoch in range(args.epochs):
    #     train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.device)
    #     val_loss, val_acc = evaluate(model, val_loader, criterion, args.device, scheduler)
    #
    #     print(
    #         f'Epoch [{epoch + 1}/{args.epochs}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
    #
    #     # 保存最佳模型
    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         torch.save(model.state_dict(), 'best_model.pth')

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('best_model.pth'))
    # 1️⃣ 随机选择一个 batch
    data_iter = iter(test_loader)
    images, labels = next(data_iter)  # 从 DataLoader 中取出一个 batch
    # 2️⃣ 在 batch 内随机取一张图片
    img = images[0].unsqueeze(0)  # 添加 batch 维度: [1, C, H, W]
    img = img.to(args.device)
    model.eval()
    # 4️⃣ 注册 hook 打印 layer4 输出 shape
    def print_shape_hook(module, input, output):
        print("layer4 output shape:", output.shape)

    handle = model.layer3.register_forward_hook(print_shape_hook)

    # 5️⃣ 前向传播
    with torch.no_grad():
        _ = model(img)

    handle.remove()

    # 测试
    # test_loss, test_acc = evaluate(model, test_loader, criterion, args.device, scheduler)
    # print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # 保存训练好的原始模型
    # torch.save(model.state_dict(), f"resnet10_{args.data_flag}.pth")
    # print(f"模型已保存到 resnet10_{args.data_flag}.pth")
    #
    # # 将训练好的原始模型中的bn层融合并保存
    # model.eval()
    # # 1. 第一层卷积（conv1） + BN 融合
    # if hasattr(model, 'bn1'):
    #     model.conv1 = fuse_conv_bn(model.conv1, model.bn1)
    #     model.bn1 = nn.Identity()
    # else:
    #     model.conv1.weight.data.copy_(model.conv1.weight.data)
    #     if model.conv1.bias is not None:
    #         model.conv1.bias.data.copy_(model.conv1.bias.data)
    # # 2. 转移残差块权重，并删除 BN
    # for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
    #     layer = getattr(model, layer_name)
    #     if hasattr(layer, 'bn1'):
    #         layer.conv1 = fuse_conv_bn(layer.conv1, layer.bn1)
    #         layer.bn1 = nn.Identity()
    #     else:
    #         layer.conv1.weight.data.copy_(layer.conv1.weight.data)
    #         if layer.conv1.bias is not None:
    #             layer.conv1.bias.data.copy_(layer.conv1.bias.data)
    #     if hasattr(layer, 'bn2'):
    #         layer.conv2 = fuse_conv_bn(layer.conv2, layer.bn2)
    #         layer.bn2 = nn.Identity()
    #     else:
    #         layer.conv2.weight.data.copy_(layer.conv2.weight.data)
    #         if layer.conv2.bias is not None:
    #             layer.conv2.bias.data.copy_(layer.conv2.bias.data)
    #
    #     # shortcut
    #     if hasattr(block, 'downsample') and block.downsample is not None:
    #         # 仅保留 conv 的权重，BN 融合
    #         conv_ds = block.downsample[0]
    #         bn_ds = block.downsample[1] if len(block.downsample) > 1 else None
    #         if bn_ds is not None:
    #             block.downsample[0] = fuse_conv_bn(conv_ds, bn_ds)
    #         else:
    #             block.downsample[0].weight.data.copy_(conv_ds.weight.data)
    #             if conv_ds.bias is not None:
    #                 block.downsample[0].bias.data.copy_(conv_ds.bias.data)
    # # 保存融合conv和bn后的模型
    # torch.save(model.state_dict(), f"resnet10_{args.data_flag}_fused.pth")
    # print(f"模型已保存到 resnet10_{args.data_flag}_fused.pth")


if __name__ == '__main__':
    main()

# Epoch [1/20] Train Loss: 0.2233, Acc: 0.9246 | Val Loss: 0.2547, Acc: 0.9245
# Epoch [2/20] Train Loss: 0.0606, Acc: 0.9791 | Val Loss: 0.2562, Acc: 0.9227
# Epoch [3/20] Train Loss: 0.0360, Acc: 0.9876 | Val Loss: 0.1327, Acc: 0.9532
# Epoch [4/20] Train Loss: 0.0265, Acc: 0.9914 | Val Loss: 0.0790, Acc: 0.9646
# Epoch [5/20] Train Loss: 0.0208, Acc: 0.9935 | Val Loss: 0.1823, Acc: 0.9422
# Epoch [6/20] Train Loss: 0.0196, Acc: 0.9941 | Val Loss: 0.1493, Acc: 0.9559
# Epoch [7/20] Train Loss: 0.0175, Acc: 0.9940 | Val Loss: 0.0736, Acc: 0.9626
# Epoch [8/20] Train Loss: 0.0111, Acc: 0.9965 | Val Loss: 0.1059, Acc: 0.9615
# Epoch [9/20] Train Loss: 0.0150, Acc: 0.9955 | Val Loss: 0.1138, Acc: 0.9576
# Epoch [10/20] Train Loss: 0.0164, Acc: 0.9947 | Val Loss: 0.0902, Acc: 0.9599
# Epoch [11/20] Train Loss: 0.0151, Acc: 0.9949 | Val Loss: 0.1174, Acc: 0.9552
# Epoch [12/20] Train Loss: 0.0113, Acc: 0.9963 | Val Loss: 0.1160, Acc: 0.9559
# Epoch [13/20] Train Loss: 0.0086, Acc: 0.9974 | Val Loss: 0.0881, Acc: 0.9598
# Epoch [14/20] Train Loss: 0.0080, Acc: 0.9974 | Val Loss: 0.1491, Acc: 0.9512
# Epoch [15/20] Train Loss: 0.0062, Acc: 0.9980 | Val Loss: 0.1137, Acc: 0.9587
# Epoch [16/20] Train Loss: 0.0087, Acc: 0.9973 | Val Loss: 0.1464, Acc: 0.9549
# Epoch [17/20] Train Loss: 0.0103, Acc: 0.9964 | Val Loss: 0.1389, Acc: 0.9559
# Epoch [18/20] Train Loss: 0.0063, Acc: 0.9982 | Val Loss: 0.1118, Acc: 0.9630
# Epoch [19/20] Train Loss: 0.0094, Acc: 0.9969 | Val Loss: 0.1619, Acc: 0.9566
# Epoch [20/20] Train Loss: 0.0115, Acc: 0.9963 | Val Loss: 0.2226, Acc: 0.9404
#
# Test Loss: 0.4934, Test Acc: 0.8962

# Epoch [1/20] Train Loss: 0.3049, Acc: 0.8989 | Val Loss: 0.1976, Acc: 0.9353
# Epoch [2/20] Train Loss: 0.0705, Acc: 0.9773 | Val Loss: 0.1488, Acc: 0.9465
# Epoch [3/20] Train Loss: 0.0393, Acc: 0.9873 | Val Loss: 0.1826, Acc: 0.9435
# Epoch [4/20] Train Loss: 0.0235, Acc: 0.9927 | Val Loss: 0.1653, Acc: 0.9458
# Epoch [5/20] Train Loss: 0.0254, Acc: 0.9920 | Val Loss: 0.1736, Acc: 0.9487
# Epoch [6/20] Train Loss: 0.0176, Acc: 0.9947 | Val Loss: 0.1889, Acc: 0.9461
# Epoch [7/20] Train Loss: 0.0164, Acc: 0.9948 | Val Loss: 0.1731, Acc: 0.9478
# Epoch [8/20] Train Loss: 0.0224, Acc: 0.9926 | Val Loss: 0.1618, Acc: 0.9524
# Epoch [9/20] Train Loss: 0.0141, Acc: 0.9953 | Val Loss: 0.1929, Acc: 0.9427
# Epoch [10/20] Train Loss: 0.0124, Acc: 0.9958 | Val Loss: 0.1945, Acc: 0.9495
# Epoch [11/20] Train Loss: 0.0040, Acc: 0.9988 | Val Loss: 0.1762, Acc: 0.9529
# Epoch [12/20] Train Loss: 0.0011, Acc: 0.9997 | Val Loss: 0.1586, Acc: 0.9555
# Epoch [13/20] Train Loss: 0.0007, Acc: 0.9999 | Val Loss: 0.1670, Acc: 0.9546
# Epoch [14/20] Train Loss: 0.0007, Acc: 0.9998 | Val Loss: 0.1701, Acc: 0.9546
# Epoch [15/20] Train Loss: 0.0007, Acc: 0.9999 | Val Loss: 0.1660, Acc: 0.9547
# Epoch [16/20] Train Loss: 0.0006, Acc: 0.9999 | Val Loss: 0.1660, Acc: 0.9536
# Epoch [17/20] Train Loss: 0.0003, Acc: 0.9999 | Val Loss: 0.1630, Acc: 0.9544
# Epoch [18/20] Train Loss: 0.0003, Acc: 1.0000 | Val Loss: 0.1589, Acc: 0.9544
# Epoch [19/20] Train Loss: 0.0003, Acc: 1.0000 | Val Loss: 0.1655, Acc: 0.9539
# Epoch [20/20] Train Loss: 0.0003, Acc: 0.9999 | Val Loss: 0.1603, Acc: 0.9544
#
# Test Loss: 0.5395, Test Acc: 0.8789
# 模型已保存到 resnet10_organamnist.pth
