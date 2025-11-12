"""
Pyfhel-based CKKS: insert padding=1 into a single ciphertext that stores an image
packed as (C, H, W) row-major into slots.

This file provides:
 - pad_ciphertext_ckks_pyfhel(HE, ct_x, C, H, W, pad=1)
   -> returns a ciphertext whose slots contain the padded image of shape
      (C, H+2*pad, W+2*pad) in row-major order.

Assumptions & notes
 - ct_x is a CKKS ciphertext produced by HE.encryptFrac(vec) where vec is the
   row-major flattening of a single image of shape (C, H, W).
 - Slots are treated circularly by rotations. We assume there are enough slots
   (usually n/2) to hold the padded layout. If not, the function will raise.
 - This implementation uses the *elementwise extraction* approach:
       1) rotate to bring the desired input element to slot 0
       2) multiply_plain by a plaintext mask that keeps only slot 0
       3) rotate to destination slot
       4) add into accumulator
   This is simple and correct but not the most efficient for large images.

API differences between Pyfhel versions:
 - Depending on Pyfhel version, the function to get slot count may differ. We
   try HE.get_nSlots(); if unavailable fall back to HE.n//2 (common).
 - rotateKeyGen: we call HE.rotateKeyGen() (no args) and assume it produces the
   needed rotation keys; if your environment requires explicit rotation indices
   you will need to generate keys for the set of rotations used.

Usage example is at the bottom.

"""

import numpy as np
from Pyfhel import Pyfhel, PyCtxt, PyPtxt


def _get_slot_count(HE: Pyfhel) -> int:
    """Return the number of CKKS slots for this HE context.
    Tries several attribute names for compatibility."""
    # Common approaches; adapt if your Pyfhel exposes a different API
    if hasattr(HE, 'get_nSlots'):
        return HE.get_nSlots()
    if hasattr(HE, 'nSlots'):
        return HE.nSlots
    if hasattr(HE, 'n'):
        return HE.n // 2
    # Last resort: try context attributes
    try:
        return HE.context.n // 2
    except Exception:
        raise RuntimeError('Cannot determine CKKS slot count from Pyfhel object.\n'
                           'Please modify _get_slot_count to match your Pyfhel API.')


def pad_ciphertext_ckks_pyfhel(HE: Pyfhel, ct_x: PyCtxt, C: int, H: int, W: int, pad: int = 1) -> PyCtxt:
    """Return a ciphertext containing the padded image (zero padding) in row-major slots.

    Parameters
    ----------
    HE : Pyfhel
        Initialized Pyfhel object with keys (public, secret optional for tests),
        and rotation keys available for the ranges used.
    ct_x : PyCtxt
        Input ciphertext packing a single image as flatten(C,H,W) in row-major order.
    C, H, W : int
        Original image shape.
    pad : int
        Amount of zero padding to add on each side (default 1).

    Returns
    -------
    PyCtxt
        Ciphertext packing the padded image of shape (C, H+2*pad, W+2*pad).

    Notes
    -----
    - This routine uses many rotate and multiply_plain operations; for large
      images it will be slow. It is meant as a clear, correct implementation.
    - For performance you can derive masks and block-extraction schemes that
      extract several elements at once.
    """

    # Derived dimensions
    Hp = H + 2 * pad
    Wp = W + 2 * pad
    out_size = C * Hp * Wp

    slot_count = _get_slot_count(HE)
    if out_size > slot_count:
        raise ValueError(f'Not enough CKKS slots: out_size={out_size} > slot_count={slot_count}')

    # Create an accumulator ciphertext initialized to zero (length slot_count)
    zero_vec = np.zeros(slot_count, dtype=float)
    ct_acc = HE.encryptFrac(zero_vec)  # ciphertext of zeros

    # Prepare a plaintext mask that keeps only slot 0 (1 at 0, 0 elsewhere)
    mask_slot0 = np.zeros(slot_count, dtype=float)
    mask_slot0[0] = 1.0
    p_mask_slot0 = HE.encodeFrac(mask_slot0)  # plaintext object

    # Optionally ensure rotation keys exist. Depending on Pyfhel version this
    # call may take an iterable of indices; many versions accept no-arg keygen.
    try:
        HE.rotateKeyGen()
    except Exception:
        # If your Pyfhel requires a list of rotations, you must generate them
        # explicitly before calling this function. We continue assuming keys exist.
        pass
    # For each channel and each (i,j) copy the single value into the correct output slot
    for c in range(C):
        for i in range(H):
            for j in range(W):
                idx_in = c * (H * W) + i * W + j
                idx_out = c * (Hp * Wp) + (i + pad) * Wp + (j + pad)
                ct_x1 = ct_x.copy()
                # 1) rotate to bring desired input element to slot 0
                HE.rotate(ct_x1, idx_in)

                # 2) multiply_plain by mask that keeps only slot 0
                if p_mask_slot0.mod_level != ct_x1.mod_level:
                    HE.mod_switch_to_next(p_mask_slot0)
                HE.multiply_plain(ct_x1, p_mask_slot0)
                HE.rescale_to_next(ct_x1)

                # 3) rotate to destination slot
                HE.rotate(ct_x1, -idx_out)

                # 4) add into accumulator
                if ct_acc.mod_level != ct_x1.mod_level:
                    HE.mod_switch_to_next(ct_acc)
                ct_acc.scale = ct_x1.scale
                ct_acc = HE.add(ct_acc, ct_x1)

    return ct_acc


# ------------------ Example usage ------------------
if __name__ == '__main__':
    # Quick self-test (small sizes). Adapt params to your secure settings.
    HE = Pyfhel()
    params = {
        'scheme': 'CKKS',
        'n': 2**14,       # play/test: use 2**16 or larger for real workloads
        'scale': 2**30,
        'qi_sizes': [40, 30, 30, 30, 30, 30, 30, 30, 30, 40]
    }
    HE.contextGen(**params)
    HE.keyGen()
    HE.rotateKeyGen()

    C, H, W = 1, 3, 3
    # plain input image (row-major)
    vec = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    print(vec)
    ct = HE.encryptFrac(vec)

    ct_padded = pad_ciphertext_ckks_pyfhel(HE, ct, C, H, W, pad=1)
    decoded = HE.decryptFrac(ct_padded)

    print('decoded (first out_size):')
    print(np.round(decoded[:C*(H+2)*(W+2)], 6).reshape(C, H+2, W+2))

    # Expected padded result for the 3x3 example:
    # [[0 0 0 0 0]
    #  [0 0 1 2 0]
    #  [0 3 4 5 0]
    #  [0 6 7 8 0]
    #  [0 0 0 0 0]]


# ------------------ Complexity & tuning advice ------------------
# Complexity:
# - The implementation performs for each input element: 2 rotates, 1 multiply_plain and 1 add.
# - Total rotates: ~2 * (C*H*W) ; multiply_plain: C*H*W ; adds: C*H*W
# - This is O(n_pixels) expensive in HE ops and thus not suitable for very
#   large images unless optimized (block-extraction, multi-slot masks).

# Practical improvements:
# 1) Extract blocks of elements at once: instead of isolating slot 0, design
#    masks that keep k consecutive slots then rotate those blocks into place.
#    This reduces number of rotations and multiplies by ~k.
# 2) Use packing choices described in HyPHEN/HYPHEN and recent RNS-CKKS papers
#    to avoid per-pixel operations; they often reserve "void slots" in advance.
# 3) Precompute and store plaintext masks for common extraction patterns to
#    avoid repeated encodings.

# Parameter suggestions (example):
# - n = 2**16 (slots=32768) for moderate images; larger n if you need more slots
# - scale = 2**30..2**40 depending on multiplicative depth required later
# - qi_sizes: choose a decomposition that supports your multiplicative depth and
#   keeps performance reasonable; e.g. [60] + [30]*k + [60]
