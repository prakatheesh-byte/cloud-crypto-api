import numpy as np

def dna_encode(arr):
    b1 = (arr >> 6) & 3
    b2 = (arr >> 4) & 3
    b3 = (arr >> 2) & 3
    b4 = arr & 3
    return np.stack([b1, b2, b3, b4], axis=1)

def dna_decode(seq):
    return (
        (seq[:,0] << 6) |
        (seq[:,1] << 4) |
        (seq[:,2] << 2) |
        seq[:,3]
    ).astype(np.uint8)

def protein_operation(seq, rounds=2):
    for r in range(rounds):
        shifts = (np.sum(seq, axis=1) + r) % 4
        for i in range(len(seq)):
            seq[i] = np.roll(seq[i], shifts[i])
    return seq

def chaotic_xor(arr, r=3.99, x0=0.7):
    x = x0
    chaos = np.zeros(len(arr), dtype=np.uint8)
    for i in range(len(arr)):
        x = r * x * (1 - x)
        chaos[i] = int(x * 256) % 256
    return np.bitwise_xor(arr, chaos)

def dna_protein_encrypt(arr):
    seq = dna_encode(arr)
    seq = protein_operation(seq)
    flat = dna_decode(seq)
    return chaotic_xor(flat)

def dna_protein_decrypt(arr):
    flat = chaotic_xor(arr)
    seq = dna_encode(flat)
    seq = protein_operation(seq)
    return dna_decode(seq)
