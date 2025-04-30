import logging
import os

from tokenizers import ByteLevelBPETokenizer
from argparse import ArgumentParser


logging.basicConfig(level="INFO")

def train(dataset_path, vocab_size, output_dir):
    tokenizer = ByteLevelBPETokenizer()

    logging.info("Start training...")
    tokenizer.train(files=[dataset_path], vocab_size=vocab_size, min_frequency=2, show_progress=True,
                                  special_tokens=[
                                    "<s>",
                                    "<pad>",
                                    "</s>",
                                    "<unk>",
                                    "<mask>"]
                                )
    
    
    logging.info("Saving trained tokenizer..,")
    os.makedirs(f"./saves/{output_dir}", exist_ok=True)
    tokenizer.save_model(f"./saves", output_dir)

    #tokenizer = RobertaTokenizerFast.from_pretrained(f"./saves/{output_dir}", max_len=512)
    #tokenizer.save_pretrained(f"./saves/{output_dir}/hf")


    #logging.info(f"Tokenizer saved into ./saves/{output_dir}/hf")

def main():

    parser = ArgumentParser(description="Параметры обучения токенизатора")

    parser.add_argument(
        "vocab_size",
        type=int,
        help="Размер словаря токенизатора"
    )

    parser.add_argument(
        "save_name",
        type=str,
        help="Имя каталога для сохранения в saves"
    )
    
    args = parser.parse_args()
    data_path = "./data/texts.txt"

    train(data_path, args.vocab_size, args.save_name)

if __name__ == "__main__":
    main()
