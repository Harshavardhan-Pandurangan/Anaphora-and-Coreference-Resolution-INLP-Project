import torch
import torch.nn as nn
import random
import tqdm

from datasets import load_dataset
# from utils import extract_corefs

class Trainer:
    def __init__(self, args, model):
        self.args = args
        self.model = model

        self.dataset = load_dataset("conll2012_ontonotesv5", "english_v12")

        self.train_data = self.dataset["train"]
        self.val_data = self.dataset["validation"]
        self.test_data = self.dataset["test"]

        print("Data loaded")
        print("Training data size:", len(self.train_data))
        print("Validation data size:", len(self.val_data))
        print("Test data size:", len(self.test_data))

        self.model = model.to(args.device)
        self.optimizer = torch.optim.Adam(params=[p for p in self.model.parameters() if p.requires_grad], lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)

    def train(self):
        # Train the model
        pass

    def evaluate(self):
        # Evaluate the model
        pass

    def predict(self):
        # Predict on new data
        pass
