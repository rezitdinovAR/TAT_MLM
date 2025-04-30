from utils import create_tokenized_dataset, compute_metrics, log_memory_usage
from trainer import Fl_Trainer

from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    DataCollatorForLanguageModeling,
    TrainingArguments
)
from threading import Thread
import torch
import logging
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Using gpu: {torch.cuda.is_available()}")

def train(tokenizer_path: str, corpus_path: str, save_path: str):
    # init our tokenizer with handle for roberta special tokens
    tokenizer = RobertaTokenizerFast(
        f"{tokenizer_path}/ver4-vocab.json",
        f"{tokenizer_path}/ver4-merges.txt"
    )

    # get tokenized dataset
    train, test = create_tokenized_dataset(corpus_path, tokenizer)

    #get config for roberta and init the model
    config = RobertaConfig(
        vocab_size=len(tokenizer.get_vocab()),
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=12,
        type_vocab_size=1,
        hidden_size=768,
        intermediate_size=3072,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        initializer_range=0.02,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    model = RobertaForMaskedLM(config)

    # specify data collator for training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,  # 15% токенов маскируются
    )

    # specify training args
    training_args = TrainingArguments(
        output_dir=save_path,
        overwrite_output_dir=False,
        num_train_epochs=5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        batch_eval_metrics=True,
        eval_on_start=False,
        save_total_limit=4,
        logging_dir=f"{save_path}/logs",
        logging_steps=10,
        report_to=["tensorboard"],
        eval_strategy="steps",
        eval_steps=5000,
        learning_rate=5e-5,
        warmup_steps=250,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        save_strategy="best",
        torch_empty_cache_steps=100,
        eval_accumulation_steps=16,
        gradient_accumulation_steps=4,
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        metric_for_best_model="f1",
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        local_rank=-1
    )

    # init the trainer
    trainer = Fl_Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # start training
    logger.info("Start training...")
    Thread(target=log_memory_usage, daemon=True).start()
    #trainer.train(resume_from_checkpoint=True)
    trainer.train()
    logger.info("Training completed, model saved")
