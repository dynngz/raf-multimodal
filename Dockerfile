FROM python:3.13

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 && \
    rm -rf /var/lib/apt/lists/*

ENV DISPLAY=:99
ENV QT_QPA_PLATFORM=offscreen

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p ./content

EXPOSE 7860

CMD ["python", "app.py"]