import argparse, os

class Parser(object):
    def getParser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--pre_train", action="store_true")
        parser.add_argument("--adv_generate", action="store_true")
        parser.add_argument("--adv_train", action="store_true")
        parser.add_argument("--infer", action="store_true")
        parser.add_argument("--test_paws", action="store_true")

        parser.add_argument("--no_common_data", action="store_true")
        parser.add_argument("--num_adv", type=int, default=500)  
        parser.add_argument("--max_num_steps", type=int, default=5)
        parser.add_argument("--max_num_candidates", type=int, default=500)   
        parser.add_argument("--advgen_batch_size", type=int, default=100)
        parser.add_argument("--query_batch_size", type=int, default=625)
        parser.add_argument("--lm_batch_size", type=int, default=32)
        parser.add_argument("--max_sent_length_diff", type=int, default=3) 
        parser.add_argument("--adv_file", type=str, default="adv.json")  
        parser.add_argument("--lm_gram", type=int, default=5)
        parser.add_argument("--task", type=str, default="classification")
        parser.add_argument("--use_dev", action="store_true")
        parser.add_argument("--lm_candidates", type=int, default=25)
        parser.add_argument("--beam_size", type=int, default=25)
        parser.add_argument("--adv_steps_per_epoch", type=int, default=1000)
        parser.add_argument("--adv_examples_per_step", type=int, default=3, help="for adversarial training")
        parser.add_argument("--use_candwords_pos", action="store_false")
        parser.add_argument("--first_attack", type=int, default=1)
        parser.add_argument("--sample_only", action="store_true")
        parser.add_argument("--adv_train_slow", action="store_true")
        parser.add_argument("--lm", type=str, default="bert")
        parser.add_argument("--textfooler_synonym_num", type=int, default=3)

        parser.add_argument("--reset_adam", action="store_true")
        parser.add_argument("--reset", action="store_true")
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--print_wrong_samples", action="store_true")
        parser.add_argument("--use_torch", action="store_true")

        parser.add_argument("--dir", type=str, default="dev")
        parser.add_argument("--display_interval", type=int, default=50)
        parser.add_argument("--fine_tune_steps", type=int, default=1000)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--data", type=str, default="qqp", choices=['qqp', 'mrpc'])
        parser.add_argument("--target", type=str, default="bert", choices=['bert', 'bimpm', 'diin'])
        parser.add_argument("--word_vector", type=str, default="data/glove.840B.300d.txt")
        parser.add_argument("--bert_model", type=str, default="bert_base")
        parser.add_argument("--cpus", type=int, default=32)

        # hyperparameters of the matching model
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--dim_word", type=int, default=300)
        parser.add_argument("--dim_char_embed", type=int, default=20)
        parser.add_argument("--dim_char_lstm", type=int, default=40)
        parser.add_argument("--dim_char_cnn", type=int, default=32)
        parser.add_argument("--num_chars", type=int, default=1500)
        parser.add_argument("--max_sent_length", type=int, default=25)
        parser.add_argument("--max_char_per_word", type=int, default=10)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--lr", type=float, default=0.0005)
        parser.add_argument("--num_labels", type=int, default=2)
        parser.add_argument("--use_exact_match", action="store_true")
        parser.add_argument("--one_way_task", action="store_true")
        parser.add_argument("--lr_decay", action="store_true")
        parser.add_argument("--lowercase", action="store_false")
        parser.add_argument("--use_sgd", action="store_true")

        # sample data
        parser.add_argument("--sample_data", action="store_true")
        parser.add_argument("--sample_num", type=int, default=50)
        parser.add_argument("--sample_label", type=int, default=0)
        parser.add_argument("--sample_output", type=str, default=None)

        return parser

def update_arguments(args):
    if args.target == "diin":
        args.no_common_data = True
        args.lowercase = False
        args.lr = 0.5
        args.batch_size = 70
    elif args.target == "bert":
        args.use_torch = True
        args.batch_size = 32
        args.max_sent_length = 128
        args.adv_examples_per_step = 2

    if args.data == "qqp":
        args.max_num_steps = 5
        if args.target == "bimpm":
            args.word_vector = "data/qqp/wordvec.txt"
    elif args.data == "mrpc":
        args.max_num_steps = 10
        args.lm_candidates = args.beam_size = 40
        args.use_candwords_pos = False
        args.max_sent_length = 128
        if args.target == "diin":
            args.batch_size = 32
            args.query_batch_size = 300
        elif args.target == "bert":
            args.batch_size = 16

    if args.adv_train:
        args.beam_size = 1 
        if not args.adv_train_slow:
            args.lm_candidates = 10

    if args.sample_only:
        args.max_num_steps = 0

    return args
