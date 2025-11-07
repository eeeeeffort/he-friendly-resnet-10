from Pyfhel import Pyfhel
import time
import numpy as np

HE = Pyfhel()
HE.contextGen(scheme='CKKS', n=2 ** 14, scale=2 ** 27, qi_sizes=[30, 27, 27, 27, 27, 27, 27, 30])
HE.keyGen()
HE.rotateKeyGen()


# Example: 2 input channels, 2 output channels
img = np.array([
    [[1, 2, 3, 1],
     [4, 5, 6, 1],
     [7, 8, 9, 1],
     [1, 1, 1, 1]],
    [[9, 8, 7, 1],
     [6, 5, 4, 1],
     [3, 2, 1, 1],
     [1, 1, 1, 1]]
], dtype=float)  # shape (2,4,4)

# Two output-channel kernels (C_out=2, C_in=2, kH=2, kW=2)
kernels = np.array([
    # out channel 0
    [
        [[1, 0],
         [-1, 2]],
        [[0, 1],
         [2, -1]]
    ],
    # out channel 1
    [
        [[2, -1],
         [0, 1]],
        [[1, 1],
         [-1, 0]]
    ]
], dtype=float)
zero_mask = np.zeros(2**13)
zero_mask[0] = 1
zero_mask = HE.encodeFrac(zero_mask)
HE.mod_switch_to_next(zero_mask)

def generate_masks_multi_out(input_shape, kernels, stride=1, padding=(0,0)):
    """
    input_shape: (C_in, H, W)
    kernels: numpy array shape (C_out, C_in, kH, kW)
    stride: int
    padding: (pad_h, pad_w)
    returns:
        masks_list: list of lists; masks_list[oc] is list of flattened masks for output channel oc
        out_H, out_W: output spatial dims
        H_p, W_p: padded input dims
    """
    C_in, H, W = input_shape
    C_out, Ck, kH, kW = kernels.shape
    assert C_in == Ck, "kernel channels must match input channels"
    pad_h, pad_w = padding
    H_p = H + 2*pad_h
    W_p = W + 2*pad_w

    out_H = (H_p - kH) // stride + 1
    out_W = (W_p - kW) // stride + 1

    # For each output channel, build list of masks
    masks_list = []
    for oc in range(C_out):
        masks_for_oc = []
        for i in range(0, out_H*stride, stride):
            for j in range(0, out_W*stride, stride):
                mask = np.zeros((C_in, H_p, W_p), dtype=float)
                for c in range(C_in):
                    for ki in range(kH):
                        for kj in range(kW):
                            rr = i + ki
                            cc = j + kj
                            mask[c, rr, cc] = kernels[oc, c, ki, kj]
                masks_for_oc.append(mask.flatten())
        masks_list.append(masks_for_oc)
    return masks_list, out_H, out_W, H_p, W_p


def simple_conv2d(img, kernels, stride=1):
    """
    简化的卷积计算，只显示结果
    """
    C_out, C_in, kH, kW = kernels.shape
    _, H, W = img.shape

    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1
    output = np.zeros((C_out, H_out, W_out))

    for out_ch in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                h_end = h_start + kH
                w_start = j * stride
                w_end = w_start + kW

                window = img[:, h_start:h_end, w_start:w_end]
                output[out_ch, i, j] = np.sum(kernels[out_ch] * window)

    return output


def print_simple_results(img, kernels, stride=1):
    """
    简洁地打印卷积结果
    """
    result = simple_conv2d(img, kernels, stride)

    print("明文卷积结果:")
    for ch in range(result.shape[0]):
        print(result[ch])

    return result


def main():
    img_shape = img.shape
    c_out = kernels.shape[0]
    img_vec = img.flatten()
    img_ctx = HE.encryptFrac(img_vec)

    masks_list, out_H, out_W, H_p, W_p = generate_masks_multi_out(img.shape, kernels, stride=1)
    in_len = img.shape[1] * img.shape[2]
    out_len = out_H * out_W

    # print(masks_list)
    res = np.zeros(2**13)
    res_index = 0
    res = HE.encryptFrac(res)
    for oc in range(c_out):
        for i, mask in enumerate(masks_list[oc]):
            # 提取对应mask进行卷积
            img_ctx_dot = img_ctx.copy()
            mask = HE.encodeFrac(mask)
            mul = HE.multiply_plain(img_ctx_dot, mask)
            HE.rescale_to_next(mul)

            # 将卷积结果求和得到对应输出位结果
            stride = in_len // 2
            mul_dot = mul.copy()
            while stride >= 1:
                HE.rotate(mul_dot, stride)
                mul = mul + mul_dot
                mul_dot = mul.copy()
                stride = stride // 2

            # 将多通道的输出求和得到当前卷积核对应输出通道的结果并将该结果添加到最终输出的相应位置
            tmp = mul.copy()
            for j in range(img_shape[0]-1):
                HE.rotate(tmp, in_len)
                mul = mul + tmp
            HE.multiply_plain(mul, zero_mask)
            HE.rescale_to_next(mul)
            HE.rotate(mul, -res_index)
            res_index += 1
            res = res + mul

    res = HE.decryptFrac(res)
    res_matrix = np.array(res[:c_out*out_len]).reshape(c_out,out_W,out_H)
    print("密文卷积结果：")
    print(res_matrix)
    # 输出明文卷积结果进行比较
    print_simple_results(img, kernels, stride=1)


if __name__ == '__main__':
    main()

