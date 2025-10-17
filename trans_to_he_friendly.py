import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO
from torchvision.models.resnet import ResNet, BasicBlock

from Pyfhel import Pyfhel, PyPtxt, PyCtxt


# 数据集
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


# 矩阵编解码、加解密函数
def encode_matrix(HE, matrix):
    # 确保为double类型
    if isinstance(matrix, np.ndarray):
        if matrix.dtype != np.float64:
            matrix = matrix.astype(np.float64)
    else:
        matrix = np.array(matrix, dtype=np.float64)
    if matrix.ndim == 0:
        # 标量，直接编码
        return HE.encodeFrac([matrix])
    elif matrix.ndim == 1:
        # 一维向量
        return HE.encodeFrac(matrix)
    else:
        # 多维矩阵，递归逐层处理
        return np.array([encode_matrix(HE, m) for m in matrix], dtype=object)


def decode_matrix(HE, matrix):
    try:
        return np.array(list(map(HE.decodeFrac, matrix)))
    except TypeError:
        return np.array([decode_matrix(HE, m) for m in matrix])


def encrypt_matrix(HE, matrix):
    try:
        return np.array(list(map(HE.encryptFrac, matrix)))
    except TypeError:
        return np.array([encrypt_matrix(HE, m) for m in matrix])


def decrypt_matrix(HE, matrix):
    try:
        return np.array(list(map(HE.decryptFrac, matrix)))
    except TypeError:
        return np.array([decrypt_matrix(HE, m) for m in matrix])


# 多项式拟合ReLu
class HeReLu:
    def __init__(self, HE):
        self.HE = HE
        print("HeReLu构造完成...")

    def __call__(self, image):
        return quadratic_poly(self.HE, image)


def quadratic_poly(HE, image):
    """
    使用二次多项式 0.25x² + 0.5x + 0.25 拟合ReLU激活函数
    在同态加密环境下计算: 0.25 * x^2 + 0.5 * x + 0.25
    """
    try:
        # 向量化实现
        return np.array(list(map(lambda x:
                                 HE.add(
                                     HE.add(
                                         HE.multiply_scalar(HE.power(x, 2), 0.25),  # 0.25 * x^2
                                         HE.multiply_scalar(x, 0.5)  # 0.5 * x
                                     ),
                                     0.25  # + 0.25
                                 ), image)))
    except TypeError:
        # 递归处理多维数组
        return np.array([quadratic_poly(HE, m) for m in image])


# 重写卷积层
class ConvolutionalLayer:
    def __init__(self, HE, weights, stride=(1, 1), padding=(0, 0), bias=None):
        self.HE = HE
        self.weights = encode_matrix(HE, weights)
        self.stride = stride
        self.padding = padding
        self.bias = bias
        if bias is not None:
            self.bias = encode_matrix(HE, bias)
        print("ConvolutionalLayer构造完成...")

    def __call__(self, t):
        t = apply_padding(t, self.padding)
        result = np.array([[np.sum([convolute2d(image_layer, filter_layer, self.stride)
                                    for image_layer, filter_layer in zip(image, _filter)], axis=0)
                            for _filter in self.weights]
                           for image in t])

        if self.bias is not None:
            return np.array([[layer + bias for layer, bias in zip(image, self.bias)] for image in result])
        else:
            return result


def convolute2d(image, filter_matrix, stride):
    x_d = len(image[0])
    y_d = len(image)
    x_f = len(filter_matrix[0])
    y_f = len(filter_matrix)

    y_stride = stride[0]
    x_stride = stride[1]

    x_o = ((x_d - x_f) // x_stride) + 1
    y_o = ((y_d - y_f) // y_stride) + 1

    def get_submatrix(matrix, x, y):
        index_row = y * y_stride
        index_column = x * x_stride
        return matrix[index_row: index_row + y_f, index_column: index_column + x_f]

    return np.array(
        [[np.sum(get_submatrix(image, x, y) * filter_matrix) for x in range(0, x_o)] for y in range(0, y_o)])


def apply_padding(t, padding):
    y_p = padding[0]
    x_p = padding[1]
    zero = t[0][0][y_p + 1][x_p + 1] - t[0][0][y_p + 1][x_p + 1]
    return [[np.pad(mat, ((y_p, y_p), (x_p, x_p)), 'constant', constant_values=zero) for mat in layer] for layer in t]


# 重写平均池化
class AveragePoolLayer:
    def __init__(self, HE, kernel_size, stride=(1, 1), padding=(0, 0)):
        self.HE = HE
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        print("AveragePoolLayer构造完成...")

    def __call__(self, t):
        t = apply_padding(t, self.padding)
        return np.array([[_avg(self.HE, layer, self.kernel_size, self.stride) for layer in image] for image in t])


def _avg(HE, image, kernel_size, stride):
    x_s = stride[1]
    y_s = stride[0]

    x_k = kernel_size[1]
    y_k = kernel_size[0]

    x_d = len(image[0])
    y_d = len(image)

    x_o = ((x_d - x_k) // x_s) + 1
    y_o = ((y_d - y_k) // y_s) + 1

    denominator = HE.encodeFrac(1 / (x_k * y_k))

    def get_submatrix(matrix, x, y):
        index_row = y * y_s
        index_column = x * x_s
        return matrix[index_row: index_row + y_k, index_column: index_column + x_k]

    return [[np.sum(get_submatrix(image, x, y)) * denominator for x in range(0, x_o)] for y in range(0, y_o)]


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


# 重写残差块 先融合bn层然后转换he-layer
class HeBasicBlock:
    def __init__(self, HE, layer=None):
        """
        同态加密残差块
        Args:
            HE: 同态加密实例
            in_planes: 输入通道数
            planes: 输出通道数
            stride: 步长
        """
        self.HE = HE
        self.shortcut_layer = None

        # 融合BN层
        print("conv1_bn1融合...")
        self.conv1 = fuse_conv_bn(layer[0].conv1, layer[0].bn1)
        print("conv2_bn2融合...")
        self.conv2 = fuse_conv_bn(layer[0].conv2, layer[0].bn2)
        if hasattr(layer[0], 'downsample') and layer[0].downsample is not None:
            # 仅保留 conv 的权重，BN 融合
            print("有下采样, conv3_bn3融合...")
            conv_ds = layer[0].downsample[0]
            bn_ds = layer[0].downsample[1] if len(layer[0].downsample) > 1 else None
            if bn_ds is not None:
                self.shortcut_layer = fuse_conv_bn(conv_ds, bn_ds)
            else:
                self.shortcut_layer.weight.data.copy_(conv_ds.weight.data)
                if conv_ds.bias is not None:
                    self.shortcut_layer.bias.data.copy_(conv_ds.bias.data)
        else:
            print("没有下采样")
        # 主路径 - 两个卷积层
        print("构造BasicBlock的conv1...")
        if self.conv1.bias is None:
            conv1_bias = None
        else:
            conv1_bias = self.conv1.bias.detach().cpu().numpy()
        self.conv1 = ConvolutionalLayer(HE, weights=self.conv1.weight.detach().numpy(),
                                        stride=self.conv1.stride,
                                        padding=self.conv1.padding,
                                        bias=conv1_bias)
        print("构造BasicBlock的conv2...")
        if self.conv2.bias is None:
            conv2_bias = None
        else:
            conv2_bias = self.conv2.bias.detach().cpu().numpy()
        self.conv2 = ConvolutionalLayer(HE, weights=self.conv2.weight.detach().numpy(),
                                        stride=self.conv2.stride,
                                        padding=self.conv2.padding,
                                        bias=conv2_bias)
        print("构造BasicBlock的shortcut...")
        # 快捷连接
        if self.shortcut_layer is not None:
            if self.shortcut_layer.bias is None:
                shortcut_layer_bias = None
            else:
                shortcut_layer_bias = self.shortcut_layer.bias.detach().cpu().numpy()
            self.shortcut_layer = ConvolutionalLayer(HE, weights=self.shortcut_layer.weight.detach().numpy(),
                                                     stride=self.shortcut_layer.stride,
                                                     padding=self.shortcut_layer.padding,
                                                     bias=shortcut_layer_bias)
        print("HeBasicBlock构造完成...")

    def __call__(self, x):
        """前向传播"""
        return self.forward(x)

    def forward(self, x):
        """
        残差块前向传播
        Args:
            x: 加密的输入张量
        Returns:
            加密的输出张量
        """
        # 主路径
        out = self.conv1(x)
        out = HeReLu(out)

        out = self.conv2(out)

        # 快捷连接
        shortcut = x
        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        # 残差连接: 主路径 + 快捷连接
        out = self.HE.add(out, shortcut)
        out = HeReLu(self.HE)(out)

        return out


# 重写全连接层
class LinearLayer:
    def __init__(self, HE, weights, bias=None):
        self.HE = HE
        self.weights = encode_matrix(HE, weights)
        self.bias = bias
        if bias is not None:
            self.bias = encode_matrix(HE, bias)
        print("LinearLayer构造完成...")

    def __call__(self, t):
        result = np.array([[np.sum(image * row) for row in self.weights] for image in t])
        if self.bias is not None:
            result = np.array([row + self.bias for row in result])
        return result


# 用于加载训练好的模型
class ResNet10(ResNet):
    def __init__(self, num_classes, in_channels=3):
        super().__init__(block=BasicBlock, layers=[1, 1, 1, 1])
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(512, num_classes)


# 将pytorch模型各层转换为he-layer类
def build_from_pytorch(HE, net):
    # 为各层创建build方法

    def conv_layer(layer):
        if layer.bias is None:
            bias = None
        else:
            bias = layer.bias.detach().numpy()

        return ConvolutionalLayer(HE, weights=layer.weight.detach().numpy(),
                                  stride=layer.stride,
                                  padding=layer.padding,
                                  bias=bias)

    def relu_layer(layer):
        return HeReLu(HE)

    def fc_layer(layer):
        if layer.bias is None:
            bias = None
        else:
            bias = layer.bias.detach().numpy()
        return LinearLayer(HE, layer.weight.detach().numpy(),
                           bias)

    def avg_pool_layer(layer):
        # This proxy is required because in PyTorch an AvgPool2d can have kernel_size, stride and padding either of
        # type (int, int) or int, unlike in Conv2d
        kernel_size = 1
        stride = (1, 1)
        padding = None

        return AveragePoolLayer(HE, kernel_size, stride, padding)

    def basic_block(layer):
        print("构造BasicBlock...")
        return HeBasicBlock(HE, layer)

    # Maps every PyTorch layer type to the correct builder
    # conv1,bn1,relu,maxpool,layer1,layer2,layer3,layer4,avgpool,fc
    options = {"co": conv_layer,
               "fc": fc_layer,
               "av": avg_pool_layer,
               "la": basic_block,
               "ma": avg_pool_layer,
               "re": relu_layer
               }

    encoded_layers = [
        options[name[:2]](layer)
        for name, layer in net.named_children()
        if name[:2] in options
    ]
    return encoded_layers


def main():
    # 初始化同态加密库
    HE = Pyfhel()
    # HE.contextGen(scheme='CKKS',
    #               n=2 ** 14,  # 多项式模度数
    #               scale=2 ** 30,  # 缩放因子
    #               qi_sizes=[60, 30, 30, 60])  # 素数位数
    HE.contextGen(scheme='CKKS', n=2 ** 13, scale=2 ** 29, qi_sizes=[60, 30, 30])
    HE.keyGen()
    HE.relinKeyGen()
    print("Pyfhel上下文生成完成！")

    # 加载原始模型
    model = ResNet10(num_classes=11, in_channels=3)
    model.load_state_dict(torch.load('resnet10_organamnist_avg_pool.pth'))
    # # 4️⃣ 注册 hook 打印 layer4 输出 shape 用于计算全局平均池化的kernel_size = (H_in // H_out, W_in // W_out)和stride = (H_in //
    # H_out, W_in // W_out)
    # def print_shape_hook(module, input, output): print("layer4 output shape:", output.shape)
    #
    # handle = model.layer4.register_forward_hook(print_shape_hook)
    #
    # # 5️⃣ 前向传播
    # with torch.no_grad():
    #     _ = model(img)
    #
    # handle.remove()

    # 将pytorch模型转换为同态加密友好类的list
    encrypt_model = build_from_pytorch(HE, model)
    for layer in encrypt_model:
        print(layer.__class__)

    # 加密一张测试集图像进行测试
    _, _, test_loader, _, _ = get_dataloaders('organamnist', 128, True)
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    sample_img = images[0]

    with torch.no_grad():
        expected_output = model(sample_img.unsqueeze(0))
    encrypted_image = encrypt_matrix(HE, sample_img.unsqueeze(0).numpy())

    start_time = time.time()
    for layer in encrypt_model:
        encrypted_image = layer(encrypted_image)
        print(f"Passed layer {layer}...")

    requested_time = round(time.time() - start_time, 2)

    result = decrypt_matrix(HE, encrypted_image)
    difference = expected_output.numpy() - result

    print(f"\nThe encrypted processing of one image requested {requested_time} seconds.")
    print(f"\nThe expected result was:")
    print(expected_output)

    print(f"\nThe actual result is: ")
    print(result)

    print(f"\nThe error is:")
    print(difference)

if __name__ == '__main__':
    main()
