from fastapi import FastAPI, UploadFile, File
import numpy as np

app = FastAPI(title="Lightweight Image Crypto API")

@app.post("/encrypt")
async def encrypt_image(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    encrypted = np.bitwise_xor(arr, 255)  # lightweight XOR
    return {
        "encrypted_bytes": encrypted.tolist(),
        "length": len(encrypted)
    }

@app.post("/decrypt")
async def decrypt_image(data: list):
    arr = np.array(data, dtype=np.uint8)
    decrypted = np.bitwise_xor(arr, 255)
    return {
        "decrypted_bytes": decrypted.tolist(),
        "length": len(decrypted)
    }
