import argparse

from classification.classifier.data_loader import load_and_cache_examples
from classification.classifier.trainer import Trainer
from classification.classifier.utils import init_logger, load_tokenizer, set_seed, plot_scores


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, train_dataset=train_dataset, test_dataset=test_dataset)

    out_filename = f"{args.task}_{args.train_file}_layers={args.n_layers}_div={args.layer_size_divisor}_d={args.dropout_rate}_lr={args.learning_rate}"

    if args.do_train:
        global_step, tr_loss, metrics = trainer.train(out_filename)
        plot_scores(metrics, args.eval_dir, out_filename)

    if args.do_eval:
        trainer.load_model(f"{out_filename}_-1")
        results = trainer.evaluate("test")
        print("Eval score:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="ant", type=str, help="The name of the task to train")
    parser.add_argument(
        "--data_dir",
        default="../dataset/ant",
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument("--model_dir", default="model/ant", type=str, help="Path to model")
    parser.add_argument(
        "--eval_dir",
        default="./eval",
        type=str,
        help="Evaluation script, result directory",
    )
    parser.add_argument("--train_file", default="train0.txt", type=str, help="Train file")
    parser.add_argument("--test_file", default="val0.txt", type=str, help="Test file")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Label file")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="EMBEDDIA/crosloengual-bert",
        help="Model Name or Path",
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation.")
    parser.add_argument(
        "--max_seq_len",
        default=500,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=2,
        type=int,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--n_layers",
        default=2,
        type=int,
        help="Number of layers following the BERT part.",
    )
    parser.add_argument(
        "--layer_size_divisor",
        default=4,
        type=int,
        help="Number by which to divide to get layer sizes.",
    )

    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--dropout_rate",
        default=0.1,
        type=float,
        help="Dropout for fully-connected layers",
    )

    parser.add_argument(
        "--early_stopping_epochs",
        type=int,
        default=10,
        help="Minimum number of epochs before early stopping.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="How many epochs can loss increase before early stopping.",
    )

    parser.add_argument(
        "--save_epochs",
        type=int,
        default=5,
        help="Save checkpoint every X epochs.",
    )

    parser.add_argument("--do_train", default=True, action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", default=False, action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--add_sep_token",
        action="store_true",
        help="Add [SEP] token at the end of the sentence",
    )

    args = parser.parse_args()

    main(args)

