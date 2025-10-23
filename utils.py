import tenseal as ts
import math
import sys

bits_scale = 28
def test_max_mults(poly_deg=2**15, coeff_mod_bit_sizes=[40] + [bits_scale]*18 + [40], global_scale=pow(2, bits_scale)):
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_deg, -1, coeff_mod_bit_sizes=coeff_mod_bit_sizes)
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    ctx.global_scale = global_scale
    ctx.auto_rescale = True

    # 加密一个简单值（单槽）
    x = 0.6
    expected = x
    enc = ts.ckks_vector(ctx, [x])

    max_mult = 0
    try:
        while True:
            # Horner: (0.25*x + 0.5) * x  --> 这里只测乘法次数：一次密文*密文
            tmp = enc * 0.25        # scalar mult
            tmp = tmp + 0.5         # add
            prod = tmp * enc        # 密文 × 密文

            # 尝试解密并检查是否接近真实值
            out = (prod).decrypt()[0]
            # 理论值：

            expected = 0.25*(expected**2) + 0.5*expected
            err = abs(out - expected)
            print(f"After mult {max_mult+1}, decrypt -> {out:.6f}, expected {expected:.6f}, err {err:.6e}")
            # 如果误差太大，认为不可用
            if err > 1e-1:  # 1e-2 threshold 可调
                print("Error too large, stopping.")
                break

            # prepare next iteration: set enc = prod? we want repeated multiplies
            enc = prod  # 下一轮用上一次的结果作为 x (最苛刻)
            max_mult += 1
            if max_mult > 50:
                break
    except Exception as e:
        print("Exception during loop:", e)

    print("max successful ciphertext×ciphertext multiplications ~", max_mult)
    return max_mult

if __name__ == "__main__":
    test_max_mults()

