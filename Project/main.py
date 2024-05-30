import numpy
import torch
import torch.nn as nn
import argparse
import os

import random
import networkx
import torchtext
from boltons.iterutils import pairwise
from data_process import *

from Classes.coreference import CoreferenceResolution
from trainer import Trainer
from data_process import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-End')
    parser.add_argument('--embed_dim', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=200, help='Hidden dimension')
    parser.add_argument('--steps', type=int, default=5, help='Number of steps')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--step_size', type=int, default=20, help='Step size')
    parser.add_argument('--gamma', type=float, default=0.005, help='Gamma')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval')
    parser.add_argument('--distance_dim', type=int, default=20, help='Distance dimension')
    parser.add_argument('--speaker_dim', type=int, default=20, help='Speaker dimension')
    parser.add_argument('--dropout_embed', type=float, default=0.5, help='Dropout embed')
    parser.add_argument('--dropout_lstm', type=float, default=0.2, help='Dropout lstm')
    parser.add_argument('--dropout_score', type=float, default=0.2, help='Dropout score')
    parser.add_argument('--dropout_span', type=float, default=0.2, help='Dropout span')
    parser.add_argument('--dropout_speaker', type=float, default=0.2, help='Dropout speaker')

    return parser.parse_args()


def main(args = None):
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cuda_available = torch.cuda.is_available()
    print('CUDA available:', cuda_available)
    args.device = torch.device('cuda' if cuda_available else 'cpu')

    train_data, val_data, test_data = load_dataset()

    model = CoreferenceResolution(args)

    trainer = Trainer(args, model)
    trainer.train()

if __name__ == '__main__':
    args = parse_args()
    main(args)