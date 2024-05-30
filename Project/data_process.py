import torch
import random
import tqdm
from datasets import load_dataset

def load_dataset():
    dataset = load_dataset("conll2012_ontonotesv5", "english_v12")

    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    print("Data loaded")
    print("Training data size:", len(train_data))
    print("Validation data size:", len(val_data))
    print("Test data size:", len(test_data))

    return train_data, val_data, test_data