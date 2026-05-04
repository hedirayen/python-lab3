FROM python:3.11

WORKDIR /app

# 🔥 Ajouter les dépendances système pour OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python3","main.py"]