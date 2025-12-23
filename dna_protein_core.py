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
        shifts = (np.sum(seq, axis=1) + p) % 4
        for i in range(len(seq)):
            seq[i] = np.roll(seq[i], shifts[i])
    return seq

def protein_inverse(seq, rounds=2):
    for p in reversed(range(rounds)):
        shifts = (np.sum(seq, axis=1) + p) % 4
        for i in range(len(seq)):
            seq[i] = np.roll(seq[i], -shifts[i])
    return seq

# ---------- CHAOS ----------
def logistic_map(n, r=3.99, x0=0.7):
    x = x0
    for _ in range(100):
        x = r * x * (1 - x)

    chaos = np.zeros(n)
    for i in range(n):
        x = r * x * (1 - x)
        chaos[i] = x

    return (np.floor(chaos * 256) % 256).astype(np.uint8)

# ---------- ENCRYPT ----------
def dna_protein_encrypt(arr):
    seq = dna_encode(arr)
    seq = protein_permute(seq)
    flat = dna_decode(seq)

    chaos = logistic_map(len(flat))
    encrypted = (flat + chaos) % 256

    return encrypted.astype(np.uint8)

# ---------- DECRYPT ----------
def dna_protein_decrypt(arr):
    chaos = logistic_map(len(arr))
    flat = (arr - chaos) % 256

    seq = dna_encode(flat)
    seq = protein_inverse(seq)

    decrypted = dna_decode(seq)
    return decrypted.astype(np.uint8)
