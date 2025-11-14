class AveragePoolingLayer:
    def __init__(self, he, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)):
        """
        参数说明：
        he: 同态加密对象
        kernel_size: 池化窗口大小，默认为(2, 2)
        stride: 池化步长，默认为(2, 2)
        padding: 填充大小，默认为(0, 0)
        """
        self.HE = he
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.kH, self.kW = self.kernel_size
        # 计算平均池化权重（固定为1/(kH*kW)）
        self.pool_weight = 1.0 / (self.kH * self.kW)
        print(f"同态平均池化层构造完成... 窗口大小: {self.kernel_size}, 步长: {self.stride}, 填充: {self.padding}")

    def __call__(self, ctx, shape):
        # 从输入形状中提取通道数和空间维度
        C_in, H_in, W_in = shape

        # 如果需要填充，应用填充操作
        if self.padding != (None, None):
            ctx = pad_ciphertext_ckks_pyfhel(self.HE, ctx, C_in, H_in, W_in, self.padding[0])
            res = HE.decryptFrac(ctx)
            print("填充后解密")
            print(res[:25].reshape(5, 5))
            H_in += 2 * self.padding[0]
            W_in += 2 * self.padding[1]

        # 计算输出空间维度
        out_H = (H_in - self.kH) // self.stride[0] + 1
        out_W = (W_in - self.kW) // self.stride[1] + 1

        # 创建平均池化的虚拟卷积核（每个输入通道对应一个输出通道，权重固定）
        # 形状: (C_out, C_in, kH, kW)，这里C_out = C_in（池化不改变通道数）
        kernels = np.zeros((C_in, C_in, self.kH, self.kW), dtype=np.float64)
        # 对角线元素设为平均权重（保持通道独立性）
        for c in range(C_in):
            kernels[c, c, :, :] = self.pool_weight
        # 调用现有的卷积函数执行平均池化操作
        ctx, _, _, _ = convolution_2d(ctx, (C_in, H_in, W_in), kernels, self.stride[0], self.padding)

        return ctx, (C_in, out_H, out_W)