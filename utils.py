from Pyfhel import Pyfhel
import numpy as np

HE = Pyfhel()
HE.contextGen(scheme='CKKS', n=2**14, scale=2**30, qi_sizes=[60, 30, 30, 60])
HE.keyGen()

data = np.array([1.2, 2.3, 3.4, 4.5], dtype=np.float64)

# ✅ 编码与加密
enc = HE.encrypt(data)  # 直接加密 numpy 数组，无需手动 encode

# ✅ 同态运算
enc_add = enc + enc
enc_mul = enc * 2.0
enc_square = enc * enc

# ✅ 解密（自动解码为 numpy 数组）
dec_add = HE.decrypt(enc_add)
dec_mul = HE.decrypt(enc_mul)
dec_square = HE.decrypt(enc_square)


print("原始数据:", data)
print("加法结果:", dec_add)
print("乘标量结果:", dec_mul)
print("乘法结果:", dec_square)
