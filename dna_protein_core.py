import numpy as np

# ---------- DNA ----------
def dna_encode(arr):
    d1 = (arr >> 6) & 3
    d2 = (arr >> 4) & 3
    d3 = (arr >> 2) & 3
    d4 = arr & 3
    return np.stack([d1, d2, d3, d4], axis=1)

def dna_decode(seq):
    return (
        (seq[:,0] << 6) |
        (seq[:,1] << 4) |
        (seq[:,2] << 2) |
        seq[:,3]
    ).astype(np.uint8)

# ---------- PROTEIN ----------
def protein_permute(seq, rounds=2):
    for p in range(rounds):
        shifts = ((np.sum(seq, axis=1) + p) % 4).astype(np.int32)
        idx = np.arange(4, dtype=np.int32)
        seq = seq[np.arange(len(seq))[:, None], (idx - shifts[:, None]) % 4]
    return seq

def protein_inverse(seq, rounds=2):
    for p in reversed(range(rounds)):
        shifts = ((np.sum(seq, axis=1) + p) % 4).astype(np.int32)
        idx = np.arange(4, dtype=np.int32)
        seq = seq[np.arange(len(seq))[:, None], (idx + shifts[:, None]) % 4]
    return seq

# ---------- CHAOS ----------
def logistic_map(n, r, x0):
    x = x0
    for _ in range(50):   # warm-up
        x = r * x * (1 - x)

    chaos = np.empty(n, dtype=np.uint8)
    for i in range(n):
        x = r * x * (1 - x)
        chaos[i] = int(x * 256) & 0xFF

    return chaos

# ---------- ENCRYPT ----------
def dna_protein_encrypt(arr, dna_rounds=1, protein_rounds=2, r=3.99, x0=0.7):
    arr = arr.astype(np.uint8)

    for _ in range(dna_rounds):
        seq = dna_encode(arr)
        seq = protein_permute(seq, protein_rounds)
        arr = dna_decode(seq)

    chaos = logistic_map(len(arr), r, x0)
    encrypted = (arr.astype(np.int16) + chaos.astype(np.int16)) % 256
    return encrypted.astype(np.uint8)

# ---------- DECRYPT ----------
def dna_protein_decrypt(arr, dna_rounds=1, protein_rounds=2, r=3.99, x0=0.7):
    chaos = logistic_map(len(arr), r, x0)
    arr = (arr.astype(np.int16) - chaos.astype(np.int16)) % 256
    arr = arr.astype(np.uint8)

    for _ in range(dna_rounds):
        seq = dna_encode(arr)
        seq = protein_inverse(seq, protein_rounds)
        arr = dna_decode(seq)

    return arr.astype(np.uint8)

