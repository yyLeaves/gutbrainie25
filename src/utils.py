SEED = 11

BATCH_SIZE = 8
NUM_TOKS = 30524
NUM_LABELS = 27
NUM_EPOCHS = 40
LR = 2e-5
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 0.01

MAX_SEQ_LEN = 512

BASE_MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
PATH_BRONZE_ANNOTATIONS = "data/raw/GutBrainIE_Full_Collection_2025/Annotations/Train/bronze_quality/json_format/train_bronze.json"
PATH_GOLD_ANNOTATIONS = "data/raw/GutBrainIE_Full_Collection_2025/Annotations/Train/gold_quality/json_format/train_gold.json"
PATH_PLATINUM_ANNOTATIONS = "data/raw/GutBrainIE_Full_Collection_2025/Annotations/Train/platinum_quality/json_format/train_platinum.json"
PATH_SILVER_ANNOTATIONS = "data/raw/GutBrainIE_Full_Collection_2025/Annotations/Train/silver_quality/json_format/train_silver.json"
PATH_DEV_ANNOTATIONS = "data/raw/GutBrainIE_Full_Collection_2025/Annotations/Dev/json_format/dev.json"
PATH_AUG_ANNOTATIONS = "notebooks/articles_dev_aug.json"
DICT_PATH_TRAIN_ANNOTATIONS = {
    "bronze": PATH_BRONZE_ANNOTATIONS,
    "gold": PATH_GOLD_ANNOTATIONS,
    "platinum": PATH_PLATINUM_ANNOTATIONS,
    "silver": PATH_SILVER_ANNOTATIONS,
    'augment': PATH_AUG_ANNOTATIONS
}
# PATH_TRAIN = "data/processed/train_dataset"
# PATH_DEV = "data/processed/eval_dataset"
# PATH_TRAIN = "data/processed/train_tokenized_dataset_no_bronze"
PATH_TRAIN = "data/processed/train_tokenized_dataset_seperate"
# PATH_TRAIN = "data/processed/train_tokenized_dataset_seperate_no_bronze"

# PATH_TRAIN = "data/processed/train_tokenized_dataset_seperate_aug_bronze"

# PATH_DEV = "data/processed/dev_tokenized_dataset_seperate"

# PATH_TRAIN = "data/processed/train_tokenized_dataset"
PATH_DEV = "data/processed/dev_tokenized_dataset"
PATH_LABEL_MAP = "data/processed/label_mappings.json"
PATH_TOKENIZER = "data/processed/tokenizer"
OUTPUT_DIR = "models/biomedbert_finetuned"
PROCESSED_DIR = "data/processed"

USE_MIXED_PRECISION = 'no'# "fp16"  # 是否使用混合精度训练

def compute_metrics(predictions, labels, label_mappings):

    import evaluate

    id2label = label_mappings["id2label"]
    label_list = [id2label[str(i)] for i in range(len(id2label))]
    
    metric = evaluate.load("seqeval")
    
    true_predictions = [[label_list[p] for p in prediction][:] for prediction in predictions]
    true_labels = [[label_list[l] for l in label][:] for label in labels]
    
    print(f"Computing metrics...")
    # print(f"true_predictions: {true_predictions[1]}")
    # print(f"true_labels: {true_labels[1]}")

    try:
        results = metric.compute(predictions=true_predictions, references=true_labels)
    except ValueError as e:
        print(f"Error during metric computation: {e}")
        return {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0}
    metrics = {
        "precision": results.get("overall_precision", 0.0),
        "recall": results.get("overall_recall", 0.0),
        "f1": results.get("overall_f1", 0.0),
        "accuracy": results.get("overall_accuracy", 0.0),
    }


    for entity_type in sorted(results.keys()):
        if entity_type.startswith("overall") or not isinstance(results[entity_type], dict):
            continue
        for metric_name in ["precision", "recall", "f1"]:
            if metric_name in results[entity_type]:
                metrics[f"{entity_type}_{metric_name}"] = results[entity_type][metric_name]

    return metrics