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
@app.post("/metrics")
async def compute_metrics(
    file: UploadFile = File(...),
    dna_rounds: int = 1,
    protein_rounds: int = 2,
    r: float = 3.99,
    x0: float = 0.7
):
    # Load image
    image = Image.open(file.file).convert("L")
    I = np.array(image, dtype=np.uint8)

    # Encrypt
    Enc = dna_protein_encrypt(
        I.flatten(),
        dna_rounds,
        protein_rounds,
        r,
        x0
    ).reshape(I.shape)

    # Decrypt
    Dec = dna_protein_decrypt(
        Enc.flatten(),
        dna_rounds,
        protein_rounds,
        r,
        x0
    ).reshape(I.shape)

    # Compute metrics
    metrics = compute_metrics_strong(
        I, Enc, Dec,
        dna_protein_encrypt,
        dna_rounds,
        protein_rounds,
        r,
        x0
    )

    return metrics
