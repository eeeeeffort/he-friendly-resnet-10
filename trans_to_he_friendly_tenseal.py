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

import tenseal as ts  # <-- 使用 TenSEAL 代替 Pyfhel

# ========== 全局 TenSEAL context（构造后赋值） ==========
TENSEAL_CONTEXT = None

# 数据集
def get_dataloaders(data_flag, batch_size=128, download=True):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    n_channels = info['n_channels']

    # 数据预处理
    if n_channels == 1:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        in_channels = 3
    else:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5] * 3, std=[.5] * 3)
        ])
        in_channels = 3

    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    val_dataset = DataClass(split='val', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader, len(info['label']), in_channels


# ---------------------
# TenSEAL 相关的编码/加解密工具
# ---------------------

def encode_matrix(HE, matrix):
    """
    TenSEAL 版本：**不对权重加密**，直接以 numpy 数组返回（保持明文）。
    仅在需要将权重以明文参与同态运算时直接使用 numpy 数组。
    """
    if isinstance(matrix, np.ndarray):
        if matrix.dtype != np.float64:
            matrix = matrix.astype(np.float64)
    else:
        matrix = np.array(matrix, dtype=np.float64)
    return matrix  # 明文返回


def encrypt_matrix(ctx, matrix):
    """
    对输入（如图片）做加密。
    - 如果是标量 -> 使用单槽 CKKS 向量 ts.ckks_vector(ctx, [val])
    - 如果是多维数组 -> 递归处理，返回嵌套列表/ndarray 结构，其叶子是 ckks_vector
    """
    if isinstance(matrix, np.ndarray):
        # 0-dim (标量)
        if matrix.ndim == 0:
            return ts.ckks_vector(ctx, [float(matrix.item())])
        # 1-dim 向量 -> 将每个元素单独加密为单槽向量（保留原结构）
        if matrix.ndim == 1:
            return np.array([ts.ckks_vector(ctx, [float(v)]) for v in matrix], dtype=object)
        # 多维 -> 递归
        return np.array([encrypt_matrix(ctx, m) for m in matrix], dtype=object)
    else:
        # 若传入的是 Python list / 标量
        try:
            arr = np.array(matrix, dtype=np.float64)
            return encrypt_matrix(ctx, arr)
        except Exception:
            # 标量 fallback
            return ts.ckks_vector(ctx, [float(matrix)])


def decrypt_matrix(ctx, matrix):
    """
    从 TenSEAL 的 ckks_vector 中解密回浮点数（标量）。
    对应 encrypt_matrix 的结构，返回 numpy 数组或标量。
    """
    if isinstance(matrix, np.ndarray) or isinstance(matrix, list):
        # 递归解密
        return np.array([decrypt_matrix(ctx, m) for m in matrix])
    else:
        # 假设 leaf 是 ts.CKKSVector
        try:
            vals = matrix.decrypt()  # 返回列表
            if isinstance(vals, (list, tuple, np.ndarray)):
                # 如果单槽向量，取第一个
                return vals[0]
            return vals
        except Exception as e:
            # 如果不是 ckks_vector，则直接返回（可能是浮点）
            return matrix


# ---------------------
# 多项式拟合 ReLU（TenSEAL 版本）
# ---------------------
class HeReLu:
    def __init__(self, ctx):
        self.ctx = ctx
        print("HeReLu（TenSEAL 版）构造完成...")

    def __call__(self, image):
        return quadratic_poly(self.ctx, image)


def quadratic_poly(ctx, image):
    """
    使用二次多项式 0.25x^2 + 0.5x + 0.25 拟合 ReLU
    采用horner式计算
    对 leaf 为 ckks_vector 的结构进行运算：
    对单个 ckks_vector x:
        x * x -> ckks_vector (elementwise)
        (x * x) * 0.25 -> ckks_vector（乘以明文标量）
        x * 0.5 -> ckks_vector
        + 0.25 -> add scalar
    """
    if isinstance(image, np.ndarray) or isinstance(image, list):
        return np.array([quadratic_poly(ctx, m) for m in image], dtype=object)
    else:
        x = image
        tmp = x * 0.25 # 乘法一次（标量乘，不增长scale）
        tmp = tmp + 0.5 # 加法，不影响scale
        tmp = tmp * x # 🔥 仅这一处是密文乘法
        enc_result = tmp + 0.25 # 加法，不影响scale
        return enc_result


# ---------------------
# 重写卷积层（权重为明文 numpy 数组）
# ---------------------
class ConvolutionalLayer:
    def __init__(self, ctx, weights, stride=(1, 1), padding=(0, 0), bias=None):
        """
        weights: numpy array with shape (out_channels, in_channels, kH, kW) -- 明文
        输入 t 的结构与加密矩阵相对应：t 为 batch 级别的嵌套列表/ndarray，
        leaf 是 ckks_vector（单槽）
        """
        self.ctx = ctx
        self.weights = encode_matrix(ctx, weights)  # 明文 numpy 数组
        self.stride = stride
        self.padding = padding
        # bias 保持明文（numpy）
        if bias is not None:
            self.bias = np.array(bias, dtype=np.float64)
        else:
            self.bias = None


    def __call__(self, t):
        # t: batch x in_channels x H x W (叶子为 ckks_vector)
        # t = apply_padding(t, self.padding, self.ctx)
        result = np.array([[np.sum([convolute2d_channel(image_layer, filter_layer, self.stride, self.ctx)
                                    for image_layer, filter_layer in zip(image, _filter)], axis=0)
                            for _filter in self.weights]
                           for image in t], dtype=object)

        if self.bias is not None:
            # bias: 明文一维数组，与 filters 对应
            return np.array([[layer + float(bias) for layer, bias in zip(image, self.bias)] for image in result],
                            dtype=object)
        else:
            return result


def convolute2d_channel(image_channel, filter_channel, stride, ctx):
    """
    image_channel: HxW with ckks_vector leafs
    filter_channel: numpy 2D filter (kH x kW) (明文)
    stride: tuple
    返回：二维数组（每个元素是 ckks_vector）
    """
    x_d = len(image_channel[0])
    y_d = len(image_channel)
    y_f = filter_channel.shape[0]
    x_f = filter_channel.shape[1]

    y_stride = stride[0]
    x_stride = stride[1]

    x_o = ((x_d - x_f) // x_stride) + 1
    y_o = ((y_d - y_f) // y_stride) + 1

    def get_submatrix(channel, x, y):
        index_row = y * y_stride
        index_col = x * x_stride
        # 返回 kH x kW 的叶子（ckks_vector）
        return [row[index_col:index_col + x_f] for row in channel[index_row:index_row + y_f]]

    out = []
    for y in range(0, y_o):
        row = []
        for x in range(0, x_o):
            sub = get_submatrix(image_channel, x, y)  # list of rows, each row is list of ckks_vector
            # 将 sub 与 filter_channel 对应位置相乘（子元素为 ckks_vector，filter 为明文标量）
            acc = None
            for i in range(y_f):
                for j in range(x_f):
                    pixel = sub[i][j]  # ckks_vector
                    w = float(filter_channel[i, j])
                    # ckks_vector * scalar 返回 ckks_vector
                    term = pixel * w
                    if acc is None:
                        acc = term
                    else:
                        acc = acc + term
            # acc 为 ckks_vector（单槽）
            row.append(acc)
        out.append(row)
    return np.array(out, dtype=object)


def apply_padding(t, padding, ctx):
    y_p = padding[0]
    x_p = padding[1]
    # 创建加密的 0（单槽）
    zero_enc = ts.ckks_vector(ctx, [0.0])
    # t: batch x channels x H x W
    padded_batch = []
    for image in t:
        padded_image = []
        for channel in image:
            H = len(channel)
            W = len(channel[0])
            # pad rows
            # pad top
            top = [[zero_enc for _ in range(W)] for _ in range(y_p)]
            bottom = [[zero_enc for _ in range(W)] for _ in range(y_p)]
            # pad each row with zeros on left/right
            new_rows = []
            for row in channel:
                left = [zero_enc for _ in range(x_p)]
                right = [zero_enc for _ in range(x_p)]
                new_row = left + list(row) + right
                new_rows.append(new_row)
            padded_channel = top + new_rows + bottom
            padded_image.append(np.array(padded_channel, dtype=object))
        padded_batch.append(np.array(padded_image, dtype=object))
    return np.array(padded_batch, dtype=object)


# ---------------------
# Average pool（TenSEAL 版）
# ---------------------
class AveragePoolLayer:
    def __init__(self, ctx, kernel_size, stride=(1, 1), padding=(0, 0)):
        self.ctx = ctx
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        print("AveragePoolLayer（TenSEAL 版）构造完成...")

    def __call__(self, t):
        # t = apply_padding(t, self.padding, self.ctx)
        return np.array([[_avg(self.ctx, layer, self.kernel_size, self.stride) for layer in image] for image in t],
                        dtype=object)


def _avg(ctx, image, kernel_size, stride):
    x_s = stride[1]
    y_s = stride[0]

    x_k = kernel_size
    y_k = kernel_size

    x_d = len(image[0])
    y_d = len(image)

    x_o = ((x_d - x_k) // x_s) + 1
    y_o = ((y_d - y_k) // y_s) + 1

    denominator = 1.0 / (x_k * y_k)

    def get_submatrix(matrix, x, y):
        index_row = y * y_s
        index_col = x * x_s
        return matrix[index_row: index_row + y_k, index_col: index_col + x_k]

    out = []
    for y in range(0, y_o):
        row = []
        for x in range(0, x_o):
            sub = get_submatrix(np.array(image, dtype=object), x, y)  # kH x kW of ckks_vector
            acc = None
            for i in range(y_k):
                for j in range(x_k):
                    term = sub[i][j] * denominator  # ckks_vector * scalar
                    if acc is None:
                        acc = term
                    else:
                        acc = acc + term
            row.append(acc)
        out.append(row)
    return np.array(out, dtype=object)


# --------------------------
# Conv + BN 融合（保持不变）
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
# HeBasicBlock（TenSEAL 版，权重为明文）
# --------------------------
class HeBasicBlock:
    def __init__(self, ctx, layer=None):
        """
        同态加密残差块
        Args:
            HE: 同态加密实例
            in_planes: 输入通道数
            planes: 输出通道数
            stride: 步长
        """
        self.ctx = ctx
        self.shortcut_layer = None

        self.conv1 = fuse_conv_bn(layer[0].conv1, layer[0].bn1)
        self.conv2 = fuse_conv_bn(layer[0].conv2, layer[0].bn2)
        if hasattr(layer[0], 'downsample') and layer[0].downsample is not None:
            conv_ds = layer[0].downsample[0]
            bn_ds = layer[0].downsample[1] if len(layer[0].downsample) > 1 else None
            if bn_ds is not None:
                self.shortcut_layer = fuse_conv_bn(conv_ds, bn_ds)
            else:
                # 若没有 bn，则直接复制 conv_ds 权重到 shortcut_layer
                tmp_conv = nn.Conv2d(conv_ds.in_channels, conv_ds.out_channels,
                                     kernel_size=conv_ds.kernel_size, stride=conv_ds.stride,
                                     padding=conv_ds.padding, bias=(conv_ds.bias is not None))
                tmp_conv.weight.data.copy_(conv_ds.weight.data)
                if conv_ds.bias is not None:
                    tmp_conv.bias.data.copy_(conv_ds.bias.data)
                self.shortcut_layer = tmp_conv

        # 主路径 - 两个卷积层
        conv1_bias = None if self.conv1.bias is None else self.conv1.bias.detach().cpu().numpy()
        self.conv1 = ConvolutionalLayer(ctx, weights=self.conv1.weight.detach().cpu().numpy(),
                                        stride=self.conv1.stride,
                                        padding=self.conv1.padding,
                                        bias=conv1_bias)
        conv2_bias = None if self.conv2.bias is None else self.conv2.bias.detach().cpu().numpy()
        self.conv2 = ConvolutionalLayer(ctx, weights=self.conv2.weight.detach().cpu().numpy(),
                                        stride=self.conv2.stride,
                                        padding=self.conv2.padding,
                                        bias=conv2_bias)
        if self.shortcut_layer is not None:
            tmp = self.shortcut_layer
            shortcut_bias = None if tmp.bias is None else tmp.bias.detach().cpu().numpy()
            self.shortcut_layer = ConvolutionalLayer(ctx, weights=tmp.weight.detach().cpu().numpy(),
                                                     stride=tmp.stride,
                                                     padding=tmp.padding,
                                                     bias=shortcut_bias)
        print("HeBasicBlock（TenSEAL 版）构造完成...")

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out = self.conv1(x)
        # out = HeReLu(self.ctx)(out)

        out = self.conv2(out)

        shortcut = x
        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        # 残差连接（两个 ckks_vector 相加）
        # out 与 shortcut 是 batch x channels x H x W 的嵌套结构
        out = add_encrypted_tensors(out, shortcut)
        out = HeReLu(self.ctx)(out)
        return out


def add_encrypted_tensors(a, b):
    """
    对两个具有相同形状的嵌套结构做逐元素相加（叶子为 ckks_vector 或明文）
    返回新的嵌套 numpy array
    """
    if isinstance(a, np.ndarray) or isinstance(a, list):
        return np.array([add_encrypted_tensors(ai, bi) for ai, bi in zip(a, b)], dtype=object)
    else:
        # leaf 应为 ckks_vector
        return a + b


# --------------------------
# LinearLayer（保持权重明文）
# --------------------------
class LinearLayer:
    def __init__(self, ctx, weights, bias=None):
        self.ctx = ctx
        self.weights = encode_matrix(ctx, weights)  # 明文 numpy (out_features x in_features)
        self.bias = None if bias is None else np.array(bias, dtype=np.float64)
        print("LinearLayer 构造完成 (权重为明文) ...")

    def __call__(self, t):
        # t: batch x features (features 为单槽 ckks_vector)
        # 我们假设 t 是二维： batch x in_features (每个元素是 ckks_vector)
        # weights: out_features x in_features (明文)
        batch_size = len(t)
        out_features = self.weights.shape[0]
        result = []
        for i in range(batch_size):
            row = []
            for j in range(out_features):
                acc = None
                for k in range(self.weights.shape[1]):
                    x = t[i][k]  # ckks_vector
                    w = float(self.weights[j, k])
                    term = x * w
                    if acc is None:
                        acc = term
                    else:
                        acc = acc + term
                if self.bias is not None:
                    acc = acc + float(self.bias[j])
                row.append(acc)
            result.append(np.array(row, dtype=object))
        return np.array(result, dtype=object)


# ResNet10 类（保持不变，注意 avgpool 使用的是 AveragePoolLayer 的占位）
class ResNet10(ResNet):
    def __init__(self, num_classes, in_channels=3):
        super().__init__(block=BasicBlock, layers=[1, 1, 1, 1])
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(512, num_classes)


# 构建 he-friendly 层列表（使用 TenSEAL context）
def build_from_pytorch(ctx, net):
    def conv_layer(layer):
        bias = None if layer.bias is None else layer.bias.detach().numpy()
        print("ConvolutionalLayer 构造完成... (权重为明文)")
        return ConvolutionalLayer(ctx, weights=layer.weight.detach().numpy(),
                                  stride=layer.stride,
                                  padding=layer.padding,
                                  bias=bias)

    def relu_layer(layer):
        return HeReLu(ctx)

    def fc_layer(layer):
        bias = None if layer.bias is None else layer.bias.detach().numpy()
        return LinearLayer(ctx, layer.weight.detach().numpy(),
                           bias)

    def avg_pool_layer(layer):
        kernel_size = 1
        stride = (1, 1)
        padding = None
        return AveragePoolLayer(ctx, kernel_size, stride, padding)

    def basic_block(layer):
        return HeBasicBlock(ctx, layer)

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


# --------------------------
# 主函数（初始化 TenSEAL，上下文，加载模型并测试单张图像）
# --------------------------
def main():
    global TENSEAL_CONTEXT
    bits_scale = 25
    # TenSEAL 上下文初始化（CKKS）
    poly_mod_degree = 2**14
    coeff_mod_bit_sizes = [40] + [bits_scale]*4 + [40] 
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1,
    coeff_mod_bit_sizes=coeff_mod_bit_sizes)
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    ctx.global_scale = 2**20
    # 为了本地可解密，我们在 context 中保留 secret key（TenSEAL 默认在创建时包含 secret key）
    TENSEAL_CONTEXT = ctx
    print("TenSEAL 上下文生成完成！")

    # 加载原始模型（与之前相同）
    model = ResNet10(num_classes=11, in_channels=3)
    model.load_state_dict(torch.load('resnet10_organamnist_avg_pool.pth', map_location='cpu'))
    model.eval()

    # 将 pytorch 模型转换为 TenSEAL-friendly 的层列表（权重明文）
    encrypt_model = build_from_pytorch(ctx, model)
    print("密文推理网络构建完成，结构如下：")
    for layer in encrypt_model:
        print(layer.__class__)

    # 加密一张测试集图像进行测试（只加密输入，权重为明文）
    _, _, test_loader, _, _ = get_dataloaders('organamnist', 128, True)
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    sample_img = images[0]

    with torch.no_grad():
        expected_output = model(sample_img.unsqueeze(0))

    # 把 sample_img 转为 numpy 并加密（结构：batch x channels x H x W）
    start_time = time.time()
    encrypted_image = encrypt_matrix(ctx, sample_img.unsqueeze(0).numpy())
    requested_time = round(time.time() - start_time, 2)
    print(f"\nThe encrypted processing of one image requested {requested_time} seconds.")
    

    start_time = time.time()
    for layer in encrypt_model:
        mid_start_time = time.time()    
        encrypted_image = layer(encrypted_image)
        mid_requested_time = round(time.time() - mid_start_time, 2)
        print(f"Passed layer {layer}...,requested {mid_requested_time} seconds.")

    requested_time = round(time.time() - start_time, 2)

    result = decrypt_matrix(ctx, encrypted_image)
    difference = expected_output.numpy() - result

    print(f"\nThe encrypted input inference processing of one image requested {requested_time} seconds.")
    print(f"\nThe expected result was:")
    print(expected_output)

    print(f"\nThe actual result is: ")
    print(result)

    print(f"\nThe error is:")
    print(difference)


if __name__ == '__main__':
    main()

