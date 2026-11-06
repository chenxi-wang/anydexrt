"""Shared argument definitions for the standalone Python scripts.
Author: Chenxi Wang, Ying Feng
"""
from pathlib import Path


def add_glove_arguments(parser, required=False):
    parser.add_argument('--glove-serial', required=required)
    parser.add_argument('--address', default='tcp://127.0.0.1:8000')
    parser.add_argument('--hand-type', choices=['left', 'right'], default='right')


def add_training_arguments(parser, epochs):
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)


def prepare_training(parser, args):
    import random
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        parser.error('Training requires a CUDA-capable PyTorch environment')
    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0:
        parser.error('epochs >= 1, batch-size >= 2, and workers >= 0 are required')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
