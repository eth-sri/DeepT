# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import torch
import argparse, os


class Parser(object):
    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()

        # modes
        parser.add_argument("--train", action="store_true")
        parser.add_argument("--train-adversarial", action="store_true")
        parser.add_argument("--infer", action="store_true")
        parser.add_argument("--verify", action="store_true")
        parser.add_argument("--attack-type", type=str, default="lp", choices=["lp", "synonym"])
        parser.add_argument("--debug-zonotope", action="store_true")
        parser.add_argument("--pgd", action="store_true")
        parser.add_argument("--word_label", action="store_true")

        # synonym attack
        parser.add_argument("--max-synonyms-per-word", type=int, default=8)
        parser.add_argument("--lirpa-ckpt", type=str, default="None")

        # data
        parser.add_argument("--dir", type=str, default="dev")
        parser.add_argument("--base-dir", type=str, default="model_base")
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--data", type=str, default="yelp",
                            choices=["yelp", "sst", "cifar", "mnist"])
        parser.add_argument("--use_tsv", action="store_true")
        parser.add_argument("--vocab_size", type=int, default=50000)
        parser.add_argument("--small", action="store_true")
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--use_dev", action="store_true")
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--task", type=str, default="text_classification", choices=["text_classification", "image"])
        parser.add_argument("--lirpa-data", action="store_true")

        # runtime
        parser.add_argument("--display_interval", type=int, default=50)

        # model
        parser.add_argument("--num-epoches", type=int, default=3)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--max_sent_length", type=int, default=128)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--num_labels", type=int, default=2)
        parser.add_argument("--num_layers", type=int, default=12)
        parser.add_argument("--num_attention_heads", type=int, default=4)
        parser.add_argument("--hidden_size", type=int, default=256)
        parser.add_argument("--intermediate_size", type=int, default=512)
        parser.add_argument("--warmup", type=float, default=-1)
        parser.add_argument("--hidden_act", type=str, default="relu")
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--min_word_freq", type=int, default=50)
        parser.add_argument("--layer_norm", type=str, default="no_var", choices=["standard", "no", "no_var"])

        # verification
        parser.add_argument("--samples", type=int, default=10)
        parser.add_argument("--p", type=int, default=2)
        parser.add_argument("--eps", type=float, default=1e-5)
        parser.add_argument("--max_eps", type=float, default=0.01)
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--debug_pos", type=int, default=1)
        parser.add_argument("--log", type=str, default="log.txt")
        parser.add_argument("--res", type=str, default="res")
        parser.add_argument("--max_verify_length", type=int, default=32)
        parser.add_argument("--method", type=str, default="forward",
                            choices=["baf", "backward", "forward",
                                     "ibp", "discrete",
                                     "forward-convex", "backward-convex", "baf-convex",
                                     "zonotope"])
        parser.add_argument("--zonotope-slow", action="store_true")
        parser.add_argument("--num_verify_iters", type=int, default=10)
        parser.add_argument("--view_embed_dist", action="store_true")
        parser.add_argument("--empty_cache", action="store_true")
        parser.add_argument("--perturbed_words", type=int, default=1, choices=[1, 2])

        # New options
        parser.add_argument("--dont_load_pretrained", action="store_true")
        parser.add_argument("--max_optim_iters", type=int, default=100)
        parser.add_argument("--discard_final_dp", action="store_true")
        parser.add_argument("--handle-one-lambda-at-the-time", action="store_true")

        # Feature flags
        parser.add_argument("--use_new_softmax", action="store_true")
        parser.add_argument("--use_new_exp", action="store_true")
        parser.add_argument("--add-softmax-sum-constraint", action="store_true")

        # Hardware
        parser.add_argument("--cpu", action="store_true")
        parser.add_argument("--cpus", type=int, default=32)
        parser.add_argument("--gpu", type=int, default=-1)
        parser.add_argument("--cpu-range", type=str, default="Default")

        # Logging
        parser.add_argument("--log-softmax", action="store_true")
        parser.add_argument("--log-error-terms-and-time", action="store_true")

        # Results
        parser.add_argument("--results-directory", type=str, default="results")

        # Error checking
        parser.add_argument("--check-zonotopes", action="store_true")

        # Zonotope error reduction
        parser.add_argument("--error-reduction-method", type=str, default="None")
        parser.add_argument("--max-num-error-terms", type=int, default=2560)

        # PGD
        parser.add_argument("--pgd-iterations", type=int, default=50)
        parser.add_argument("--eps-step-ratio", type=float, default=0.25)
        parser.add_argument("--num-pgd-starts", type=int, default=50)

        # Verification details
        parser.add_argument("--start-word", type=int, default=1)
        parser.add_argument("--start-sentence", type=int, default=1)
        parser.add_argument("--sentence-to-skip", type=int, default=-1)
        parser.add_argument("--one-word-per-sentence", action="store_true")
        parser.add_argument("--hardcoded-max-eps", action="store_true")
        parser.add_argument("--with-relu-in-pooling", action="store_true")
        parser.add_argument("--use-other-dot-product-ordering", action="store_true")
        parser.add_argument("--all-words", action="store_true")

        # Variants and efficiency
        parser.add_argument("--batch-softmax-computation", action="store_true")
        parser.add_argument("--use-dot-product-variant3", action="store_true")

        # Variant12 / Variant21
        parser.add_argument("--num-fast-dot-product-layers-due-to-switch", type=int, default=-1)
        parser.add_argument("--max-num-error-terms-fast-layers", type=int, default=-1)
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--variant2plus1", action="store_true")
        group.add_argument("--variant1plus2", action="store_true")
        
        # Gurobi parameters in constraint solving for softmax
        parser.add_argument("--timeout", type=float, default=0.01)
        parser.add_argument("--num-processes", type=int, default=12)

        parser.add_argument("--compute-synonym-region-using-solver", action="store_true")

        # Input
        parser.add_argument("--num-input-error-terms", type=int, default=128)

        # DiffAI
        parser.add_argument("--diffai", action="store_true")
        parser.add_argument("--diffai-eps", type=float, default=0.01)
        parser.add_argument("--keep-intermediate-zonotopes", action="store_true")

        # ViT Verification
        parser.add_argument("--concretize-special-norm-error-together", action="store_true")

        return parser


def update_arguments(args):
    if args.num_fast_dot_product_layers_due_to_switch != -1:
        assert args.max_num_error_terms_fast_layers != -1, "Must explicitly specify max error terms for fast layers too"
        assert args.variant2plus1 or args.variant1plus2, f"For num_fast_dot_product_layers_due_to_switch={args.num_fast_dot_product_layers_due_to_switch} pass either --variant2plus1 or --variant2plus1"
        assert args.variant2plus1 != args.variant1plus2, "Pick one of the two variants" # Paranoia code

    if "big" in args.dir:
        args.num_input_error_terms = 256
        print("For smaller network, setting args.num_input_error_terms = 256")
    elif "smaller" in args.dir or "word_small" in args.dir:
        args.num_input_error_terms = 64
        print("For smaller network, setting args.num_input_error_terms = 64")
    else:
        assert args.num_input_error_terms == 128, "num_input_error_terms: update this logic to make it work"

    if args.attack_type == "synonym":
        assert args.p > 10 or args.p == 2 or args.p == 1, f"For a synonym attack, the norm must be 1, 2 or inf (p > 10) instead of p={args.p}"
        args.eps = -1
        args.num_perturbed_words = 10000
    else:
        args.num_perturbed_words = 1

    if args.infer or (args.verify and args.attack_type != "synonym") or args.word_label or args.debug_zonotope or args.pgd:
        args.small = True

    if args.verify and args.attack_type == "synonym":
        args.samples = 50

    if args.method == 'backward-convex':
        args.discard_final_dp = True

    if not args.train:
        args.batch_size *= 30

    if args.cpu_range != "Default":
        start, end = args.cpu_range.strip().split("-")
        args.num_processes = int(end) - int(start) + 1

    if args.cpu:
        args.device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.diffai:
        args.keep_intermediate_zonotopes = True

    return args
