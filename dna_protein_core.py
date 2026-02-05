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
        (seq[:, 0] << 6) |
        (seq[:, 1] << 4) |
        (seq[:, 2] << 2) |
        seq[:, 3]
    ).astype(np.uint8)

# ---------- PROTEIN ----------
def protein_permute(seq, rounds=2):
    for p in range(rounds):
        shifts = ((np.sum(seq, axis=1) + p) % 4).astype(np.int32)
        idx = np.arange(4)
        seq = seq[np.arange(len(seq))[:, None], (idx - shifts[:, None]) % 4]
    return seq

def protein_inverse(seq, rounds=2):
    for p in reversed(range(rounds)):
        shifts = ((np.sum(seq, axis=1) + p) % 4).astype(np.int32)
        idx = np.arange(4)
        seq = seq[np.arange(len(seq))[:, None], (idx + shifts[:, None]) % 4]
    return seq

# ---------- CHAOS ----------
def logistic_map(n, r, x0):
    x = x0
    for _ in range(50):
        x = r * x * (1 - x)

    chaos = np.empty(n, dtype=np.uint8)
    for i in range(n):
        x = r * x * (1 - x)
        chaos[i] = int(x * 256) & 0xFF
    return chaos

# ---------- CHAOTIC PERMUTATION ----------
def chaotic_permutation(n, r, x0):
    x = x0
    for _ in range(100):
        x = r * x * (1 - x)

    perm = np.arange(n)
    for i in range(n):
        x = r * x * (1 - x)
        j = int(x * n) % n
        perm[i], perm[j] = perm[j], perm[i]
    return perm

def inverse_permutation(perm):
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv

# ---------- ENCRYPT ----------
def dna_protein_encrypt(arr, dna_rounds=1, protein_rounds=2, r=3.99, x0=0.7):
    arr = arr.astype(np.uint8)

    # DNA + Protein
    for _ in range(dna_rounds):
        seq = dna_encode(arr)
        seq = protein_permute(seq, protein_rounds)
        arr = dna_decode(seq)

    # 🔥 Spatial scrambling (FIX)
    perm = chaotic_permutation(len(arr), r, x0)
    arr = arr[perm]

    # Diffusion
    chaos = logistic_map(len(arr), r, x0)
    encrypted = chain_diffusion_keyed(arr, chaos, dna_rounds, protein_rounds)
    return encrypted


# ---------- DECRYPT ----------
def dna_protein_decrypt(arr, dna_rounds=1, protein_rounds=2, r=3.99, x0=0.7):
    # Reverse diffusion
    chaos = logistic_map(len(arr), r, x0)
    arr = inverse_chain_diffusion_keyed(arr, chaos, dna_rounds, protein_rounds)

    # Reverse scrambling
    perm = chaotic_permutation(len(arr), r, x0)
    inv_perm = inverse_permutation(perm)
    arr = arr[inv_perm]

    # Reverse DNA + Protein
    for _ in range(dna_rounds):
        seq = dna_encode(arr)
        seq = protein_inverse(seq, protein_rounds)
        arr = dna_decode(seq)

    return arr
def chain_diffusion_keyed(arr, chaos, dna_rounds, protein_rounds):
    n = len(arr)
    out = np.zeros_like(arr, dtype=np.int16)

    key_mix = (dna_rounds + 2 * protein_rounds) % 256

    # Forward diffusion
    prev = int(chaos[0])
    for i in range(n):
        out[i] = (int(arr[i]) + prev + int(chaos[i]) + key_mix) % 256
        prev = out[i]

    # Backward diffusion
    prev = int(chaos[-1])
    for i in range(n - 1, -1, -1):
        out[i] = (out[i] + prev + int(chaos[i]) + key_mix) % 256
        prev = out[i]

    return out.astype(np.uint8)

def inverse_chain_diffusion_keyed(arr, chaos, dna_rounds, protein_rounds):
    n = len(arr)
    out = arr.astype(np.int16)

    key_mix = (dna_rounds + 2 * protein_rounds) % 256

    # Reverse backward diffusion
    prev = chaos[-1]
    for i in range(n - 1, -1, -1):
        temp = out[i]
        out[i] = (out[i] - prev - chaos[i] - key_mix) % 256
        prev = temp

    # Reverse forward diffusion
    prev = chaos[0]
    for i in range(n):
        temp = out[i]
        out[i] = (out[i] - prev - chaos[i] - key_mix) % 256
        prev = temp

    return out.astype(np.uint8)

