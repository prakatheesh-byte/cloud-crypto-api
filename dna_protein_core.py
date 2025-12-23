import numpy as np

# ================= DNA + PROTEIN + CHAOS ENCRYPT =================
def dna_protein_encrypt(arr: np.ndarray):

    # ---- DNA encoding (2-bit) ----
    d1 = (arr >> 6) & 3
    d2 = (arr >> 4) & 3
    d3 = (arr >> 2) & 3
    d4 = arr & 3

    seq = np.stack([d1, d2, d3, d4], axis=1)

    # ---- Protein-like circular shifts ----
    shifts = np.sum(seq, axis=1) % 4
    for i in range(len(seq)):
        seq[i] = np.roll(seq[i], shifts[i])

    # ---- Logistic chaos ----
    x = 0.7
    r = 3.99
    chaos = np.zeros(len(seq))
    for i in range(len(seq)):
        x = r * x * (1 - x)
        chaos[i] = x

    chaos_bytes = (chaos * 255).astype(np.uint8)

    # ---- Combine DNA back ----
    flat = (
        seq[:, 0] * 64 +
        seq[:, 1] * 16 +
        seq[:, 2] * 4 +
        seq[:, 3]
    ).astype(np.uint8)

    encrypted = np.bitwise_xor(flat, chaos_bytes)
    return encrypted


# ================= DNA + PROTEIN + CHAOS DECRYPT =================
def dna_protein_decrypt(arr: np.ndarray):

    # ---- Regenerate same chaos ----
    x = 0.7
    r = 3.99
    chaos = np.zeros(len(arr))
    for i in range(len(arr)):
        x = r * x * (1 - x)
        chaos[i] = x

    chaos_bytes = (chaos * 255).astype(np.uint8)

    # ---- Reverse XOR ----
    decrypted = np.bitwise_xor(arr, chaos_bytes)
    return decrypted
