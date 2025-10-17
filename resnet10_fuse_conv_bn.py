import torch
import torch.nn as nn
import trans_to_he_friendly
from torchvision.models.resnet import ResNet, BasicBlock
from copy import deepcopy



# --------------------------
# ResNet-10
# --------------------------
class ResNet10(ResNet):
    def __init__(self, num_classes, in_channels=3, dropout_p=0.5):
        super().__init__(block=BasicBlock, layers=[1, 1, 1, 1])
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # 添加 Dropout 层（用于全连接层前）
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(512, num_classes)

class HE_ResNet10(ResNet):
    def __init__(self, num_classes, in_channels=3):
        super().__init__(block=BasicBlock, layers=[1, 1, 1, 1], norm_layer=nn.Identity)
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=True)
        # 替换 MaxPool -> AvgPool
        self.maxpool = nn.AvgPool2d(kernel_size=2, stride=2)
        # 全连接层
        self.fc = nn.Linear(512, num_classes)


# --------------------------
# Conv + BN 融合函数
# --------------------------
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


# --------------------------
# 自动转换函数
# --------------------------
def convert_resnet10_to_he(model, num_classes=10, in_channels=3):
    """
    model: 已训练的标准 ResNet-10
    返回: HE-friendly ResNet-10
    """
    he_model = HE_ResNet10(num_classes=num_classes, in_channels=in_channels)

    # 1. 第一层卷积（conv1） + BN 融合
    if hasattr(model, 'bn1'):
        he_model.conv1 = fuse_conv_bn(model.conv1, model.bn1)
    else:
        he_model.conv1.weight.data.copy_(model.conv1.weight.data)
        if model.conv1.bias is not None:
            he_model.conv1.bias.data.copy_(model.conv1.bias.data)

    # 2. 转移残差块权重，并删除 BN，替换 ReLU
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        old_layer = getattr(model, layer_name)
        new_layer = getattr(he_model, layer_name)
        for old_block, new_block in zip(old_layer, new_layer):
            # 融合 block 内 conv1 + bn1
            if hasattr(old_block, 'bn1'):
                new_block.conv1 = fuse_conv_bn(old_block.conv1, old_block.bn1)
            else:
                new_block.conv1.weight.data.copy_(old_block.conv1.weight.data)
                if old_block.conv1.bias is not None:
                    new_block.conv1.bias.data.copy_(old_block.conv1.bias.data)
            # 融合 block 内 conv2 + bn2
            if hasattr(old_block, 'bn2'):
                new_block.conv2 = fuse_conv_bn(old_block.conv2, old_block.bn2)
            else:
                new_block.conv2.weight.data.copy_(old_block.conv2.weight.data)
                if old_block.conv2.bias is not None:
                    new_block.conv2.bias.data.copy_(old_block.conv2.bias.data)
            # shortcut
            if hasattr(old_block, 'shortcut') and old_block.downsample is not None:
                # 仅保留 conv 的权重，BN 融合
                conv_ds = old_block.downsample[0]
                bn_ds = old_block.downsample[1] if len(old_block.downsample) > 1 else None
                if bn_ds is not None:
                    new_block.downsample[0] = fuse_conv_bn(conv_ds, bn_ds)
                else:
                    new_block.downsample[0].weight.data.copy_(conv_ds.weight.data)
                    if conv_ds.bias is not None:
                        new_block.downsample[0].bias.data.copy_(conv_ds.bias.data)

    # 3. 全连接层直接复制
    he_model.fc.weight.data.copy_(model.fc.weight.data)
    he_model.fc.bias.data.copy_(model.fc.bias.data)

    return he_model


# --------------------------
# 示例：转换已训练模型
# --------------------------
if __name__ == "__main__":
    # 训练好的 resnet10_pathmnist.pth
    state_dict = torch.load("resnet10_organamnist_1.pth", map_location='cpu')

    # 构建一个ResNet-10（用于加载权重）
    from torchvision.models.resnet import ResNet, BasicBlock

    num_classes = 11  # 根据数据集调整
    model_std = ResNet(BasicBlock, [1, 1, 1, 1])

    num_features = model_std.fc.in_features
    model_std.fc = nn.Linear(num_features, 11)
    model_std.load_state_dict(state_dict)
    model_std.eval()

    # 转换为 HE-friendly
    he_model = convert_resnet10_to_he(model_std, num_classes=num_classes, in_channels=3)

    # 保存 HE-friendly 模型
    torch.save(he_model.state_dict(), "resnet10_he_pathmnist.pth")
    print("HE-friendly ResNet-10 已保存为 resnet10_he_pathmnist.pth")
