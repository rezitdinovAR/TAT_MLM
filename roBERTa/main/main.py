from train_mlm import train

from argparse import ArgumentParser
from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    parser = ArgumentParser(description="Параметры обучения токенизатора")

    parser.add_argument(
        "tokenizer_path",
        type=str,
        help="Путь к токенайзеру"
    )

    parser.add_argument(
        "corpus_path",
        type=str,
        help="Путь к текстовому корпусу"
    )
    
    parser.add_argument(
            "save_path",
            type=str,
            help="Путь к каталогу для сохранения обученной модели"
    )

    args = parser.parse_args()
    train(args.tokenizer_path, args.corpus_path, args.save_path)


if __name__ == "__main__":
    main()