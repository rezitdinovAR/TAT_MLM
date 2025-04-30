FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 as base


RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*


RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1


COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu122


WORKDIR /main

# docker build -t mlm_trainer .
# docker run --gpus '"device=4,5,6"' --name roberta --memory="32g" \
#     --memory-swap="40g" \
#     --shm-size=8g \
#     --ulimit memlock=-1 \
#     -v ~/projects/punc_model/LSTM_roBERTa/roBERTa/main:/main/code \
#     -v ~/projects/punc_model/LSTM_roBERTa/tokenizer/saves:/main/tokenizer \
#     -v ~/projects/punct_model/data/prepared_data:/main/data \
#     -v ./saves_fl:/main/model \
#     -v ./tokenized_data:/main/tokenized_data \
#     -v ./default_config.yaml:/root/.cache/huggingface/accelerate/default_config.yaml \
#     -p 6006:6006 \
#     -d mlm_trainer
# docker exec -d roberta tensorboard --logdir ./model/logs --host 0.0.0.0 --port 6006

# accelerate launch ./code/main.py ./tokenizer/ver4 ./data ./model
# python ./code/main.py ./tokenizer/ver4 ./data ./model
CMD ["accelerate", "launch", "./code/main.py", "./tokenizer/ver4", "./data", "./model"]
