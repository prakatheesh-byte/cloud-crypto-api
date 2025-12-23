from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import os

from dna_protein_core import dna_protein_encrypt, dna_protein_decrypt


app = FastAPI(title="Lightweight Image Crypto API")

@app.post("/encrypt")
async def encrypt_image(file: UploadFile = File(...)):

    # 1. Open image as grayscale
    image = Image.open(file.file).convert("L")
    arr = np.array(image, dtype=np.uint8)

    # 2. Encrypt
    flat = arr.flatten()
    encrypted = dna_protein_encrypt(flat.astype(np.int16)).astype(np.uint8)

    # 3. Convert back to image
    encrypted_img = encrypted.reshape(arr.shape)
    enc_image = Image.fromarray(encrypted_img.astype(np.uint8))

    # 4. Save encrypted image
    save_path = "encrypted.png"
    enc_image.save(save_path)

    # 5. Return encrypted image file
    return FileResponse(
        path=save_path,
        media_type="image/png",
        filename="encrypted.png"
    )

@app.post("/decrypt")
async def decrypt_image(file: UploadFile = File(...)):

    # 1. Open encrypted image
    image = Image.open(file.file).convert("L")
    arr = np.array(image, dtype=np.uint8)

    # 2. Decrypt
    flat = arr.flatten()
    decrypted = dna_protein_decrypt(flat.astype(np.int16)).astype(np.uint8)

    # 3. Convert back to image
    decrypted_img = decrypted.reshape(arr.shape)
    dec_image = Image.fromarray(decrypted_img)

    # 4. Save decrypted image
    save_path = "decrypted.png"
    dec_image.save(save_path)

    # 5. Return decrypted image
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
