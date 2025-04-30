from transformers import RobertaConfig, RobertaTokenizerFast
from datasets import Dataset, load_from_disk
from numpy import argmax, int32
import torch
import evaluate
import logging
import psutil
import GPUtil
import os
import time


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_memory_usage(interval=25):
    while True:
        # CPU память
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss / (1024 ** 3)
        
        # GPU память
        gpu_mem = []
        GPUs = GPUtil.getGPUs()
        for gpu in GPUs:
            gpu_mem.append(f"{gpu.memoryUsed:.1f}/{gpu.memoryTotal:.1f} GB")
        
        logger.info(f"Memory Usage - CPU: {cpu_mem:.2f} GB | GPU: {', '.join(gpu_mem)}")
        
        time.sleep(interval)

def get_corpus(data_path: str):
    with open(data_path + "/train.txt", "r", encoding="utf-8") as f:
        train = [line.strip() for line in f if line.strip()]
    
    with open(data_path + "/dev.txt", "r", encoding="utf-8") as f:
        test = [line.strip() for line in f if line.strip()]
    
    return train, test
    
def tokenize(examples: Dataset, tokenizer: RobertaTokenizerFast):
    return tokenizer(examples["text"],
                     truncation=True,
                     max_length=512,
                     return_overflowing_tokens=True,
                     padding="max_length",
                     stride=64
                    )
    
def create_tokenized_dataset(data_path: str, tokenizer: RobertaTokenizerFast):
    train_path = "/main/tokenized_data/train"
    test_path = "/main/tokenized_data/test"

    if not (len(os.listdir(train_path)) > 0 and len(os.listdir(test_path)) > 0):
        logger.info("Tokenized data not found, processing raw datasets...")
        train_corpus, test_corpus = get_corpus(data_path)
        
        train_dataset = Dataset.from_dict({"text": train_corpus})
        test_dataset = Dataset.from_dict({"text": test_corpus})

        tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=["text"], fn_kwargs={"tokenizer": tokenizer})
        tokenized_test = test_dataset.map(tokenize, batched=True, remove_columns=["text"], fn_kwargs={"tokenizer": tokenizer})

        tokenized_train.save_to_disk(train_path)
        tokenized_test.save_to_disk(test_path)
    
    else:
        logger.info("Successfully loaded tokenized datasets")
        tokenized_train = load_from_disk(train_path)
        tokenized_test = load_from_disk(test_path)

    return tokenized_train, tokenized_test


def compute_metrics(eval_pred, compute_result):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    logits, labels = eval_pred
    logits = logits.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    predictions = argmax(logits, axis=-1)

    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask].astype(int32)  # Конвертация в int32
    valid_labels = labels[valid_mask].astype(int32)            # Конвертация в int32

    if len(valid_labels) == 0:
        return {"accuracy": 0.0, "f1": 0.0}

    accuracy_results = accuracy.compute(
        predictions=valid_predictions,
        references=valid_labels
    )
    
    f1_results = f1.compute(
        predictions=valid_predictions,
        references=valid_labels,
        average="macro"
    )
    
    return {
        "accuracy": accuracy_results["accuracy"],
        "f1": f1_results["f1"]
    }
