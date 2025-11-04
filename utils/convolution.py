from Pyfhel import Pyfhel
import numpy as np

img = np.array([
    [1., 2., 3., 4.],
    [0., 1., 0., 2.],
    [1., 0., 1., 1.],
    [2., 1., 0., 0.]
], dtype=np.float64)

kernel = np.array([
    [1., 0.],
    [-1., 1.]
], dtype=np.float64)

HE = Pyfhel()
HE.contextGen(scheme='CKKS', n=2 ** 14, scale=2 ** 27, qi_sizes=[30, 27, 27, 27, 27, 27, 27, 30])
HE.keyGen()
HE.rotateKeyGen()
padded_len = 16


# ===================================================
# 自动生成明文掩码函数
# ===================================================
def generate_conv_masks(input_shape, kernel, stride=1):
    """
    根据输入尺寸、卷积核、步长生成卷积掩码列表。
    input_shape: (H, W)
    kernel: numpy array of shape (kH, kW)
    stride: 步长
    return: list of masks (each mask is flattened length H*W)
    """
    H, W = input_shape
    kH, kW = kernel.shape
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1

    masks = []
    for i in range(0, out_H * stride, stride):
        for j in range(0, out_W * stride, stride):
            mask = np.zeros((H, W), dtype=float)
            for ki in range(kH):
                for kj in range(kW):
                    mask[i + ki, j + kj] = kernel[ki, kj]
            masks.append(mask.flatten())
    return masks


def main():
    ptx_img = HE.encodeFrac(img.flatten())
    ctx_img = HE.encryptPtxt(ptx_img)

    masks = generate_conv_masks(img.shape, kernel, stride=1)
    res = [0]*padded_len
    res = HE.encryptFrac(np.array(res, dtype=np.float64))

    for i, mask in enumerate(masks):
        ctx_img_tmp = ctx_img.copy()
        ptx_mask = HE.encodeFrac(mask)
        mul = HE.multiply_plain(ctx_img_tmp, ptx_mask)
        HE.rescale_to_next(mul)
        stride = padded_len // 2

        mul_dot = mul.copy()
        while stride >= 1:
            HE.rotate(mul_dot, stride)
            mul = mul + mul_dot
            mul_dot = mul.copy()
            stride = stride // 2
        HE.rotate(mul, -i)
        zero_mask = [0]*i + [1] + (padded_len-i-1)*[0]
        zero_mask = HE.encodeFrac(np.array(zero_mask, dtype=np.float64))
        HE.mod_switch_to_next(zero_mask)
        HE.multiply_plain(mul, zero_mask)
        HE.rescale_to_next(mul)
        res = res + mul

    res = HE.decryptFrac(res)
    res_matrix = np.array(res[:9]).reshape(3, 3)
    print(res_matrix)


if __name__ == '__main__':
    main()
