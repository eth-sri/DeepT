import os
# Make it possible to setup the process_range in software
import sys

import pickle
import psutil

from Logger import Logger
from Parser import Parser, update_arguments

argv = sys.argv[1:]
parser = Parser.get_parser()
args, _ = parser.parse_known_args(argv)

args = update_arguments(args)
args.with_lirpa_transformer = False

# Setting up the GPU
if args.gpu != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Settin up the CPU
if psutil.cpu_count() > 4 and args.cpu_range != "Default":
    start, end = int(args.cpu_range.split("-")[0]), int(args.cpu_range.split("-")[1])
    os.sched_setaffinity(0, {i for i in range(start, end + 1)})

from data_utils import set_seeds
from Verifiers.VerifierZonotopeDNN import VerifierZonotopeDNN

# Load dataset
import mnist_loader as ml

# Load dataset
set_seeds(args.seed)
valset = ml.load_single_digits('val', [1, 7], batch_size=16, shuffle=False)

# Load model
NETWORK_NAME = '17_mnist_small.pkl'
MNIST_DIM = 784
layer_sizes = [MNIST_DIM, 10, 50, 10, 2]

network = pickle.load(open(NETWORK_NAME, 'rb'))
net = network.net
net = net.cuda()
print("Loaded pretrained network")

# Setup
import tensorflow as tf

tf1 = tf.compat.v1

if tf.__version__[0] == '2':
    tf1.disable_eager_execution()

sess = tf1.Session()

# Run it
with sess.as_default():
    args.p = 2
    logger = Logger(sess, args, ["Random string"], 1)
    verifier = VerifierZonotopeDNN(args, network, logger)
    verifier.run(valset)
