import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def image_entropy(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0,256), density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def correlation_coeff(img, direction="H"):
    img = img.astype(np.float64)

    if direction == "H":
        return np.corrcoef(img[:, :-1].flatten(), img[:, 1:].flatten())[0, 1]
    elif direction == "V":
        return np.corrcoef(img[:-1, :].flatten(), img[1:, :].flatten())[0, 1]
    elif direction == "D":
        return np.corrcoef(img[:-1, :-1].flatten(), img[1:, 1:].flatten())[0, 1]


def compute_npcr_uaci(enc1, enc2):
    diff = enc1 != enc2
    npcr = np.sum(diff) / diff.size * 100

    uaci = np.sum(np.abs(enc1.astype(np.int16) - enc2.astype(np.int16))) \
           / (enc1.size * 255) * 100

    return npcr, uaci


def compute_metrics_strong(I_gray, Enc, Dec, encrypt_func,
                           dna_rounds, protein_rounds, r, x0):

    metrics = {}

    # MSE
    metrics["MSE_enc"] = np.mean((I_gray.astype(float) - Enc.astype(float))**2)
    metrics["MSE_dec"] = np.mean((I_gray.astype(float) - Dec.astype(float))**2)

    # PSNR
    metrics["PSNR_enc"] = peak_signal_noise_ratio(I_gray, Enc, data_range=255)

    # SSIM
    metrics["SSIM"] = structural_similarity(I_gray, Enc, data_range=255)

    # Entropy
    metrics["Entropy_Orig"] = image_entropy(I_gray)
    metrics["Entropy_Enc"]  = image_entropy(Enc)

    # Correlation (Original)
    metrics["Corr_H_Orig"] = correlation_coeff(I_gray, "H")
    metrics["Corr_V_Orig"] = correlation_coeff(I_gray, "V")
    metrics["Corr_D_Orig"] = correlation_coeff(I_gray, "D")

    # Correlation (Encrypted)
    metrics["Corr_H_Enc"] = correlation_coeff(Enc, "H")
    metrics["Corr_V_Enc"] = correlation_coeff(Enc, "V")
    metrics["Corr_D_Enc"] = correlation_coeff(Enc, "D")

    # NPCR & UACI
    I_mod = I_gray.copy()
    h, w = I_gray.shape
    I_mod[h//2, w//2] ^= 128

    Enc_mod = encrypt_func(
        I_mod.flatten(),
        dna_rounds,
        protein_rounds,
        r,
        x0
    ).reshape(I_gray.shape)

    metrics["NPCR_pct"], metrics["UACI_pct"] = compute_npcr_uaci(Enc, Enc_mod)

    return metrics
