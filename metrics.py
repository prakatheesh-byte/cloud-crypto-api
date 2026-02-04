import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ---------- ENTROPY ----------
def image_entropy(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0,256), density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


# ---------- CORRELATION ----------
def correlation_coeff(img, direction="H"):
    img = img.astype(np.float64)

    if direction == "H":
        x = img[:, :-1].flatten()
        y = img[:, 1:].flatten()
    elif direction == "V":
        x = img[:-1, :].flatten()
        y = img[1:, :].flatten()
    else:
        x = img[:-1, :-1].flatten()
        y = img[1:, 1:].flatten()

    num = np.sum((x - x.mean()) * (y - y.mean()))
    den = np.sqrt(np.sum((x - x.mean())**2) * np.sum((y - y.mean())**2))

    return num / den if den != 0 else 0


# ---------- NPCR & UACI ----------
def compute_npcr_uaci(enc1, enc2):
    diff = enc1 != enc2
    npcr = np.sum(diff) / diff.size * 100

    uaci = np.sum(
        np.abs(enc1.astype(np.int16) - enc2.astype(np.int16))
    ) / (enc1.size * 255) * 100

    return npcr, uaci


# ---------- COMPLETE METRICS ----------
def compute_metrics(
    I,
    Enc,
    Dec,
    encrypt_func,
    dna_rounds,
    protein_rounds,
    r,
    x0
):
    metrics = {}

    # Encryption metrics
    metrics["MSE_enc"] = np.mean((I.astype(float) - Enc.astype(float))**2)
    metrics["PSNR_enc"] = peak_signal_noise_ratio(I, Enc, data_range=255)
    metrics["SSIM_enc"] = structural_similarity(I, Enc, data_range=255)

    # Entropy
    metrics["Entropy_Orig"] = image_entropy(I)
    metrics["Entropy_Enc"] = image_entropy(Enc)

    # Correlation
    metrics["Corr_H_Orig"] = correlation_coeff(I, "H")
    metrics["Corr_V_Orig"] = correlation_coeff(I, "V")
    metrics["Corr_D_Orig"] = correlation_coeff(I, "D")

    metrics["Corr_H_Enc"] = correlation_coeff(Enc, "H")
    metrics["Corr_V_Enc"] = correlation_coeff(Enc, "V")
    metrics["Corr_D_Enc"] = correlation_coeff(Enc, "D")

    # Decryption correctness
    metrics["MSE_dec_check"] = np.mean((I.astype(float) - Dec.astype(float))**2)

    # NPCR & UACI (computed internally)
    I_mod = I.copy()
    h, w = I.shape
    I_mod[h//2, w//2] ^= 128

    Enc_mod = encrypt_func(
        I_mod.flatten(),
        dna_rounds,
        protein_rounds,
        r,
        x0
    ).reshape(I.shape)

    metrics["NPCR_pct"], metrics["UACI_pct"] = compute_npcr_uaci(Enc, Enc_mod)

    return metrics
