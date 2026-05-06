FROM python:3.11-slim
WORKDIR /app

# Installation des dépendances système pour OpenCV (CORRIGÉE)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py image.jpg ./

# Créer une image par défaut si image.jpg n'existe pas
RUN python -c "import cv2, numpy as np; img = np.random.randint(0, 255, (512,512), dtype=np.uint8); cv2.imwrite('image.jpg', img)"

CMD ["python", "main.py"]