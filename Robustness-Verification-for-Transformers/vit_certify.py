import os
import sys

import psutil
import torch

from Parser import Parser, update_arguments
from Verifiers.VerifierZonotopeViT import VerifierZonotopeViT, sample_correct_samples
from fake_logger import FakeLogger
from mnist import mnist_test_dataloader, normalizer
from vit import ViT
from vit_attack import pgd_attack

argv = sys.argv[1:]
parser = Parser.get_parser()
args, _ = parser.parse_known_args(argv)
args = update_arguments(args)
args.with_lirpa_transformer = False  # For compatibility
args.all_words = True  # All words (i.e. patches) should be perturbed
args.concretize_special_norm_error_together = True  # The norm constraints are on the whole image
args.num_input_error_terms = 28 * 28  # The norm constraints are on the whole image

# Setting up the GPU
if args.gpu != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Settin up the CPU
if psutil.cpu_count() > 4 and args.cpu_range != "Default":
    start, end = int(args.cpu_range.split("-")[0]), int(args.cpu_range.split("-")[1])
    os.sched_setaffinity(0, {i for i in range(start, end + 1)})


from Logger import Logger  # noqa
from data_utils import set_seeds  # noqa

set_seeds(args.seed)

test_data = mnist_test_dataloader(batch_size=1, shuffle=True)

set_seeds(args.seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
#             dim=64, depth=3, heads=4, mlp_dim=128, layer_norm_type="no_var").to(device)
model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=64, depth=1, heads=4, mlp_dim=128, layer_norm_type="no_var").to(device)

model.load_state_dict(torch.load("mnist_transformer.pt"))
model.eval()

print(args.cpu)
print(args.device)
print(args)
print(f"Test Dataset size: {len(test_data)}")

logger = FakeLogger()

print("\n")
data_normalized = []
for i, (x, y) in enumerate(test_data):
    data_normalized.append({
        "label": y.to(device),
        "image": x.to(device)
    })
    if i == 100:
        break

run_pgd = args.pgd
if run_pgd:
    args.num_pgd_starts = 10
    args.pgd_iterations = 50
    args.max_eps = 2.0
    examples = sample_correct_samples(args, data_normalized, model)
    pgd_attack(model, args, examples, normalizer)
else:
    args.samples = 100
    verifier = VerifierZonotopeViT(args, model, logger, num_classes=10, normalizer=normalizer)
    verifier.run(data_normalized)
