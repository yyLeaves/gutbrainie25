import json
import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from utils import PATH_TRAIN, PATH_DEV, PATH_LABEL_MAP, BATCH_SIZE

def collate_fn(batch):
    collated_batch = {
        "input_ids": torch.stack([torch.tensor(example["input_ids"]) for example in batch]),
        "attention_mask": torch.stack([torch.tensor(example["attention_mask"]) for example in batch]),
        "labels": torch.stack([torch.tensor(example["labels"]) for example in batch]),
    }
    return collated_batch

def get_label_mappings():
    with open(PATH_LABEL_MAP, "r") as f:
        label_mappings = json.load(f)
        print(f"# Labels: {label_mappings['num_labels']}")
    return label_mappings

def get_dataloader():
    train_dataset = load_from_disk(PATH_TRAIN)
    eval_dataset = load_from_disk(PATH_DEV)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_dataloader, eval_dataloader

def get_class_weights(dataset, label_mappings):
    label_counts = np.zeros(label_mappings["num_labels"])
    for example in dataset:
        labels = example["labels"]
        for label in labels:
            if label != -100: 
                label_counts[label] += 1
    label_counts = np.maximum(label_counts, 1)
    weights = 1.0 / np.sqrt(label_counts)
    weights = weights / np.mean(weights)
    class_weights = torch.tensor(weights, dtype=torch.float)
    return class_weights

def prepare_data_for_training():
    train_dataloader, eval_dataloader = get_dataloader()
    print(f"Train set size: {len(train_dataloader.dataset)}")
    print(f"Eval set size: {len(eval_dataloader.dataset)}")
    label_mappings = get_label_mappings()
    class_weights = get_class_weights(train_dataloader.dataset, label_mappings)
    print(f'{label_mappings}')
    print(f"class weights: {class_weights}")
    return train_dataloader, eval_dataloader, label_mappings, class_weights

if __name__ == "__main__":
    prepare_data_for_training()