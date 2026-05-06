# app.py
from flask import Flask, jsonify
import cv2
import numpy as np
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio as psnr
import base64
import io
from PIL import Image

app = Flask(__name__)

# Vos fonctions existantes
def dct2(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def idct2(img):
    return idct(idct(img.T, norm='ortho').T, norm='ortho')

def generate_watermark(size=10):
    return np.random.randint(0, 2, size)

def qim_embed_blocks(dct_img, watermark, delta=150, block_size=5):
    coeffs_t = dct_img.copy()
    for i in range(len(watermark)):
        start_r, start_c = 50 + (i * 10), 50 + (i * 10)
        for r in range(start_r, start_r + block_size):
            for c in range(start_c, start_c + block_size):
                val = coeffs_t[r, c]
                if watermark[i] == 0:
                    coeffs_t[r, c] = np.round(val / delta) * delta
                else:
                    coeffs_t[r, c] = (np.round((val - delta/2) / delta) * delta) + delta/2
    return coeffs_t

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
        extracted.append(1 if sum(votes) > (len(votes)/2) else 0)
    return np.array(extracted)

@app.route('/')
def home():
    return jsonify({
        'message': 'API de Tatouage d\'Image QIM',
        'endpoints': {
            '/health': 'Vérifier l\'état du service',
            '/process': 'Exécuter le tatouage sur image.jpg'
        }
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'watermark-qim'})

@app.route('/process')
def process():
    try:
        # Charger l'image
        image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({'error': 'Image non trouvée'}), 404
        
        image = cv2.resize(image, (512, 512))
        
        # Générer watermark
        watermark = generate_watermark()
        
        # Insérer
        dct_image = dct2(image.astype(np.float64))
        watermarked_dct = qim_embed_blocks(dct_image, watermark, delta=40, block_size=5)
        watermarked_image = np.clip(idct2(watermarked_dct), 0, 255).astype('uint8')
        
        # Extraire
        extracted = qim_extract_blocks(dct2(watermarked_image.astype(float)), len(watermark), delta=40, block_size=5)
        
        # Calculer PSNR
        score_psnr = psnr(image, watermarked_image)
        
        # Convertir l'image tatouée en base64
        _, buffer = cv2.imencode('.png', watermarked_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'psnr': f"{score_psnr:.2f} dB",
            'watermark_original': watermark.tolist(),
            'watermark_extracted': extracted.tolist(),
            'ber': float(np.mean(watermark != extracted)),
            'image_base64': img_base64[:100] + '...'  # Tronqué pour l'affichage
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)