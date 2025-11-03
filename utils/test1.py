from Pyfhel import Pyfhel
import numpy as np

vec1 = [1, 2, 3, 4, 5]
vec2 = [2, 2, 2, 2, 2]

HE = Pyfhel()
HE.contextGen(scheme='CKKS', n=2 ** 14, scale=2 ** 27, qi_sizes=[40, 27, 27, 27, 27, 27, 27, 40])
HE.keyGen()
HE.rotateKeyGen()
HE.relinKeyGen()

ptx_1 = HE.encodeFrac(np.array(vec1, dtype=np.float64))
ptx_2 = HE.encodeFrac(np.array(vec2, dtype=np.float64))
ctx_1 = HE.encryptPtxt(ptx_1)
ctx_2 = HE.encryptPtxt(ptx_2)
n = max(len(vec1), len(vec2))
padded_len = 2 ** int(np.ceil(np.log2(n)))


def pad_to_power_of_two(vec, target_length):
    return vec + [0] * (target_length - len(vec))


vec1_pad = pad_to_power_of_two(vec1, padded_len)
vec2_pad = pad_to_power_of_two(vec2, padded_len)

mul = ctx_1 * ctx_2
HE.relinearize(mul)
HE.rescale_to_next(mul)
stride = padded_len // 2

mul_dot = mul.copy()
while stride >= 1:
    HE.rotate(mul_dot, stride)
    mul = mul + mul_dot
    mul_dot = mul.copy()
    stride = stride // 2

res = HE.decryptPtxt(mul)
res = HE.decodeFrac(res)
print(res)