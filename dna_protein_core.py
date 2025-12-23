import numpy as np

# ---------------- DNA ENCODING ----------------
def dna_encode(arr):
    d1 = (arr >> 6) & 3
    d2 = (arr >> 4) & 3
    d3 = (arr >> 2) & 3
    d4 = arr & 3
    return np.stack([d1, d2, d3, d4], axis=1)

# ---------------- PROTEIN ENCODING ----------------
def protein_encode(seq, rounds):
    seq = seq.copy()
    for p in range(1, rounds + 1):
        shifts = (seq.sum(axis=1) + p) % 4
        for s in range(4):
            idx = shifts == s
            if np.any(idx):
                seq[idx] = np.roll(seq[idx], shift=s, axis=1)
    return seq

# ---------------- CHAOTIC SEQUENCE ----------------
def logistic_chaos(n, r=3.99, x0=0.7):
    x = x0
    for _ in range(500):   # transient removal
        x = r * x * (1 - x)

    chaos = np.zeros(n)
    for i in range(n):
        x = r * x * (1 - x)
        chaos[i] = x

    return (np.floor(chaos * 256) % 256).astype(np.uint8)

# ---------------- FULL ENCRYPTION ----------------
def dna_protein_encrypt(arr, D=2, P=2, r=3.99):
    flat = arr.flatten().astype(np.uint8)

    # DNA encoding
    seq = dna_encode(flat)

    # DNA rounds
    for _ in range(1, D):
        seq = seq ^ 3

    # Protein rounds
    seq = protein_encode(seq, P)

    # Recombine DNA symbols
    combined = (
        seq[:, 0] * 64 +
        seq[:, 1] * 16 +
        seq[:, 2] * 4  +
        seq[:, 3]
    ).astype(np.uint8)

    # Chaos
    chaos = logistic_chaos(len(combined), r)

    # XOR diffusion
    encrypted = combined ^ chaos
    return encrypted
