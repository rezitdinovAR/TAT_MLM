FROM python:3.10-slim


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


WORKDIR /app
COPY train_bpe.py /app/train_bpe.py

# docker run --name bpe -v ~/projects/punct_model/data/prepared_data/texts.txt:/app/data/texts.txt -v ./saves:/app/saves -d tokenizer_trainer
CMD ["python", "train_bpe.py", "50265", "ver4"]
