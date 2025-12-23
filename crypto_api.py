from fastapi import FastAPI, UploadFile, File
import numpy as np

# IMPORTS MUST BE AT TOP LEVEL
from dna_protein_core import dna_protein_encrypt, dna_protein_decrypt

app = FastAPI(title="Lightweight Image Crypto API")


@app.post("/encrypt")
async def encrypt_image(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)

    encrypted = dna_protein_encrypt(arr)

    return {
        "encrypted_bytes": encrypted.tolist(),
        "length": len(encrypted)
    }


@app.post("/decrypt")
async def decrypt_image(data: list):
    arr = np.array(data, dtype=np.uint8)

    decrypted = dna_protein_decrypt(arr)

    return {
        "decrypted_bytes": decrypted.tolist(),
        "length": len(decrypted)
    }


@app.get("/")
def root():
    return {
        "status": "Cloud Crypto API is running",
        "endpoints": ["/encrypt", "/decrypt"]
    }
