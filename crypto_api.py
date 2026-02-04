from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import os
from metrics import compute_metrics_strong

from dna_protein_core import dna_protein_encrypt, dna_protein_decrypt


app = FastAPI(title="Lightweight Image Crypto API")

@app.post("/encrypt")
async def encrypt_image(
    file: UploadFile = File(...),
    dna_rounds: int = 1,
    protein_rounds: int = 2,
    r: float = 3.99,
    x0: float = 0.7
):

    image = Image.open(file.file).convert("L")
    arr = np.array(image, dtype=np.uint8)

    flat = arr.flatten()
    encrypted = dna_protein_encrypt(
        flat,
        dna_rounds,
        protein_rounds,
        r,
        x0
    )

    encrypted_img = encrypted.reshape(arr.shape)
    enc_image = Image.fromarray(encrypted_img.astype(np.uint8))

    save_path = "encrypted.png"
    enc_image.save(save_path)

    return FileResponse(
        path=save_path,
        media_type="image/png",
        filename="encrypted.png"
    )

@app.post("/decrypt")
async def decrypt_image(
    file: UploadFile = File(...),
    dna_rounds: int = 1,
    protein_rounds: int = 2,
    r: float = 3.99,
    x0: float = 0.7
):

    image = Image.open(file.file).convert("L")
    arr = np.array(image, dtype=np.uint8)

    flat = arr.flatten()
    decrypted = dna_protein_decrypt(
        flat,
        dna_rounds,
        protein_rounds,
        r,
        x0
    )

    decrypted_img = decrypted.reshape(arr.shape)
    dec_image = Image.fromarray(decrypted_img.astype(np.uint8))

    save_path = "decrypted.png"
    dec_image.save(save_path)

    return FileResponse(
        path=save_path,
        media_type="image/png",
        filename="decrypted.png"
    )

@app.get("/")
def root():
    return {
        "status": "Cloud Crypto API is running",
        "endpoints": ["/encrypt", "/decrypt"]
    }
@app.post("/metrics_manual")
async def compute_metrics_manual(
    original: UploadFile = File(...),
    encrypted: UploadFile = File(...),
    decrypted: UploadFile = File(...)
):
    # Load images
    I = np.array(Image.open(original.file).convert("L"), dtype=np.uint8)
    Enc = np.array(Image.open(encrypted.file).convert("L"), dtype=np.uint8)
    Dec = np.array(Image.open(decrypted.file).convert("L"), dtype=np.uint8)

    # Shape check (IMPORTANT)
    if I.shape != Enc.shape or I.shape != Dec.shape:
        return {"error": "Image dimensions do not match"}

    # Compute metrics
    metrics = {}

    metrics["MSE_enc"] = np.mean((I.astype(float) - Enc.astype(float))**2)
    metrics["MSE_dec"] = np.mean((I.astype(float) - Dec.astype(float))**2)

    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    metrics["PSNR_enc"] = peak_signal_noise_ratio(I, Enc, data_range=255)
    metrics["SSIM"] = structural_similarity(I, Enc, data_range=255)

    # Entropy
    def entropy(img):
        hist, _ = np.histogram(img.flatten(), bins=256, range=(0,256), density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    metrics["Entropy_Orig"] = entropy(I)
    metrics["Entropy_Enc"] = entropy(Enc)

    # Correlation (MATLAB-style)
    def corr2(a, b):
        a = a - np.mean(a)
        b = b - np.mean(b)
        return np.sum(a * b) / np.sqrt(np.sum(a*a) * np.sum(b*b))

    metrics["Corr_H_Orig"] = corr2(I[:, :-1], I[:, 1:])
    metrics["Corr_V_Orig"] = corr2(I[:-1, :], I[1:, :])
    metrics["Corr_D_Orig"] = corr2(I[:-1, :-1], I[1:, 1:])

    metrics["Corr_H_Enc"] = corr2(Enc[:, :-1], Enc[:, 1:])
    metrics["Corr_V_Enc"] = corr2(Enc[:-1, :], Enc[1:, :])
    metrics["Corr_D_Enc"] = corr2(Enc[:-1, :-1], Enc[1:, 1:])

    # NPCR & UACI (between encrypted and decrypted is NOT valid)
    # NPCR/UACI should be done between Enc and Enc_mod
    metrics["NOTE"] = "NPCR/UACI must be computed using two encrypted images with slight plaintext difference"

    return metrics
