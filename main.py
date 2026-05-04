import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio as psnr

# =========================
# 1. CHARGER IMAGE
# =========================
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Erreur : image non trouvée")
    exit()

image = cv2.resize(image, (512, 512)) # Redimensionnement pour la stabilité
print("Image chargée et redimensionnée")

# =========================
# 2. FONCTIONS DCT / IDCT
# =========================
def dct2(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def idct2(img):
    return idct(idct(img.T, norm='ortho').T, norm='ortho')

# =========================
# 3. WATERMARK
# =========================
# Nous gardons ta fonction mais avec une taille fixe pour la robustesse par bloc
def generate_watermark(size=10):
    return np.random.randint(0, 2, size) # Message stable de test

watermark = generate_watermark()
print("Watermark prêt")

# =========================
# 4. QIM INSERTION (CORRIGÉE)
# =========================
def qim_embed_blocks(dct_img, watermark, delta=150, block_size=5):
    coeffs_t = dct_img.copy()
    for i in range(len(watermark)):
        # On utilise la zone de sécurité (50, 50)
        start_r, start_c = 50 + (i * 10), 50 + (i * 10)
        for r in range(start_r, start_r + block_size):
            for c in range(start_c, start_c + block_size):
                val = coeffs_t[r, c]
                if watermark[i] == 0:
                    coeffs_t[r, c] = np.round(val / delta) * delta
                else:
                    coeffs_t[r, c] = (np.round((val - delta/2) / delta) * delta) + delta/2
    return coeffs_t

delta_val = 40
b_size = 5
dct_image = dct2(image.astype(np.float64))
watermarked_dct = qim_embed_blocks(dct_image, watermark, delta=delta_val, block_size=b_size)

# =========================
# 5. IMAGE TATOUÉE
# =========================
watermarked_image = np.clip(idct2(watermarked_dct), 0, 255).astype('uint8')

# =========================
# 6. EXTRACTION (CORRIGÉE PAR VOTE)
# =========================
def qim_extract_blocks(dct_img, size, delta=150, block_size=5):
    extracted = []
    for i in range(size):
        start_r, start_c = 50 + (i * 10), 50 + (i * 10)
        votes = []
        for r in range(start_r, start_r + block_size):
            for c in range(start_c, start_c + block_size):
                val = dct_img[r, c]
                q0 = np.round(val / delta) * delta
                votes.append(0 if abs(val - q0) < delta/4 else 1)
        # Vote majoritaire pour contrer le bruit
        extracted.append(1 if sum(votes) > (len(votes)/2) else 0)
    return np.array(extracted)

# Extraction directe (sans attaque)
extracted_watermark = qim_extract_blocks(dct2(watermarked_image.astype(float)), len(watermark), delta=delta_val, block_size=b_size)

# =========================
# 7. BRUIT (ATTACK)
# =========================
def add_noise(img):
    noise = np.random.normal(0, 15, img.shape)
    noisy = img.astype(float) + noise
    return np.clip(noisy, 0, 255).astype('uint8')

noisy_image = add_noise(watermarked_image)
extracted_noisy = qim_extract_blocks(dct2(noisy_image.astype(float)), len(watermark), delta=delta_val, block_size=b_size)

# =========================
# 8. JPEG COMPRESSION
# =========================
cv2.imwrite("compressed.jpg", watermarked_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
compressed = cv2.imread("compressed.jpg", cv2.IMREAD_GRAYSCALE)
compressed = cv2.resize(compressed, (512, 512))
extracted_compressed = qim_extract_blocks(dct2(compressed.astype(float)), len(watermark), delta=delta_val, block_size=b_size)

# =========================
# 9. BER & PSNR
# =========================
def compute_ber(original, extracted):
    return np.mean(original != extracted)

ber_direct = compute_ber(watermark, extracted_watermark)
ber_noise = compute_ber(watermark, extracted_noisy)
ber_jpeg = compute_ber(watermark, extracted_compressed)
score_psnr = psnr(image, watermarked_image)

# =========================
# 11. AFFICHAGE
# =========================
plt.figure(figsize=(12,8))
imgs = [image, watermarked_image, noisy_image, compressed]
titles = ["Original", f"Tatouée (PSNR: {score_psnr:.2f}dB)", f"Bruit (BER: {ber_noise*100:.1f}%)", f"JPEG (BER: {ber_jpeg*100:.1f}%)"]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(imgs[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# =========================
# 12. RESULTATS FINAUX
# =========================
print("\n===== RESULTATS FINAUX =====")
print(f"PSNR           : {score_psnr:.2f} dB")
print(f"BER Direct     : {ber_direct*100:.2f} %")
print(f"BER Bruit      : {ber_noise*100:.2f} %")
print(f"BER JPEG (50%) : {ber_jpeg*100:.2f} %")
print("-" * 30)
print("Original :", watermark)
print("Extrait  :", extracted_compressed)