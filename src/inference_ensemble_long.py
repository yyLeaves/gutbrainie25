import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset

from pathlib import Path
import argparse
from model_with_crf import BertCrfForTokenClassification
from utils import BASE_MODEL_NAME, PATH_LABEL_MAP, OUTPUT_DIR, NUM_LABELS, MAX_SEQ_LEN, BATCH_SIZE, NUM_TOKS
from tokenize_data import InferenceDataset,InferenceDatasetCut, prepare_json_to_df
from transformers import set_seed
from torchcrf import CRF
set_seed(42)

MODEL_CHECKPOINT = "models/biomedbert_finetuned/best_model"
# LIST_MODEL_CHECKPOINT = [
#     'models/biomedbert_finetuned/best_model_7595',
#     'models/biomedbert_finetuned/best_model_7671',
#     'models/biomedbert_finetuned/best_model7705',
# ] # green
# LIST_MODEL_CHECKPOINT = [
#     'models/biomedbert_finetuned/best_model_7569',
#     'models/biomedbert_finetuned/best_model_7498',
#     'models/biomedbert_finetuned/best_model7724',
# ] # purple

LIST_MODEL_CHECKPOINT = [
        'models/biomedbert_finetuned/best_model_7569',
    'models/biomedbert_finetuned/best_model_7498',
    'models/biomedbert_finetuned/best_model7724',
    # 'models/biomedbert_finetuned/best_model_7595',
    # 'models/biomedbert_finetuned/best_model_7671',
    # 'models/biomedbert_finetuned/best_model7705',
    # 'models/biomedbert_finetuned/best_model_7506',
    # 'models/biomedbert_finetuned/best_model_7639',
    # 'models/biomedbert_finetuned/best_model_7676',
]
# MODEL_CHECKPOINT = "models/biomedbert_finetuned/baseline/best_model"

PATH_EVAL_ARTICLES = "data/raw/GutBrainIE_Full_Collection_2025/Articles/json_format/articles_test.json"

with open(PATH_LABEL_MAP, "r") as f:
    label_mappings = json.load(f)
    label2id = label_mappings['label2id']
    id2label = label_mappings['id2label']

def load_model_from_checkpoint(model_checkpoint_path, device):
    tokenizer = AutoTokenizer.from_pretrained(str(model_checkpoint_path))
    print(f"Loading model structure {BASE_MODEL_NAME} with {NUM_LABELS} labels...")
    model = BertCrfForTokenClassification(
        model_name_or_path=BASE_MODEL_NAME, # Use base name for initial structure
        num_labels=NUM_LABELS
    )
    model.resize_token_embeddings(len(tokenizer))

    base_model_state_dict = AutoModel.from_pretrained(str(model_checkpoint_path)).state_dict()
    model.bert.load_state_dict(base_model_state_dict, strict=False) # Use strict=False initially if needed
    print("Base BERT model weights loaded.")

    classifier_crf_state_path = model_checkpoint_path / "classifier_crf_state_dict.bin"
    custom_states = torch.load(classifier_crf_state_path, map_location='cpu')
    model.classifier.load_state_dict(custom_states['classifier_state_dict'])
    model.crf.load_state_dict(custom_states['crf_state_dict'])
    print("Classifier and CRF state dicts loaded.")

    model.to(device)
    print(f"Model loaded and moved to {device}.")
    return model

def load_data_for_inference(data_path):
    dataset = torch.load(data_path)
    print(f"Data loaded from {data_path}.")
    return dataset

def collate_fn_inference(batch):
    """Collate function for inference DataLoader."""
    # Separate metadata from tensors
    doc_ids = [item['doc_id'] for item in batch]
    titles = [item['title'] for item in batch]
    abstracts = [item['abstract'] for item in batch]
    offset_mappings = [item['offset_mapping'] for item in batch] # List of lists

    # Stack tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])

    return {
    "doc_ids": doc_ids,
    "title": titles,
    "abstract": abstracts,
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "offset_mappings": offset_mappings
    }

def collate_fn_inference_extra(batch):
    """Collate function for inference DataLoader."""
    # Separate metadata from tensors
    doc_ids = [item['doc_id'] for item in batch]
    titles = [item['title'] for item in batch]
    abstracts = [item['abstract'] for item in batch]
    offset_mappings = [item['offset_mapping'] for item in batch] # List of lists
    period_char_idxs = [item['period_char_idx'] for item in batch]
    # Stack tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])

    return {
    "doc_ids": doc_ids,
    "title": titles,
    "abstract": abstracts,
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "offset_mappings": offset_mappings,
    "period_char_idxs": period_char_idxs
    }

def decode_predictions(predicted_label_ids_batch, tokenizer, input_ids, attention_mask, offset_mapping, text, location):
    """Decode predictions to labels."""
    decoded_labels = []
    total_length = int(sum(attention_mask))-2 # Exclude [CLS] and [SEP] tokens
    input_ids = input_ids[2:][:total_length] # Remove [CLS] and [SEP] tokens
    predicted_label_ids_batch = predicted_label_ids_batch[2:][:total_length] 
    offset_mapping = offset_mapping[1:][:total_length]

    # Convert IDs to BIO labels
    bio_labels = [id2label.get(str(label_id), "O") for label_id in predicted_label_ids_batch]
    list_entities = []
    current_entity = None
    for i, label in enumerate(bio_labels):
        if label.startswith("B-"):
            current_entity = {
                "label": label[2:],  # Remove "B-",
                "start_idx": offset_mapping[i][0],
                "end_idx": offset_mapping[i][1]
            }
        elif label.startswith("I-") and current_entity is not None and label[2:] == current_entity["label"]:
            current_entity["end_idx"] = offset_mapping[i][1]  # Update end offset

        else:
            if current_entity is not None:
                current_entity["text_span"] = text[current_entity["start_idx"]:current_entity["end_idx"]]
                list_entities.append(current_entity)
                current_entity = None
    if current_entity is not None:
        current_entity["text_span"] = text[current_entity["start_idx"]:current_entity["end_idx"]]
        list_entities.append(current_entity)
    for entity in list_entities:
        entity["end_idx"] = entity["end_idx"] - 1 # Adjust end offset to be inclusive
    for entity in list_entities:
        entity["location"] = location
    return list_entities

def ensemble_model_decode(models, input_ids, attention_mask):
    emissions_list = []
    transitions_list = []

    for model in models:
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_masks, return_dict=True)
        emissions = outputs['logits'].detach()  # Detach to avoid gradient flow
        transitions = model.crf.transitions.detach()  # Detach transition matrix

        emissions_list.append(emissions)
        transitions_list.append(transitions)

    # Average emissions and transitions
    avg_emissions = torch.mean(torch.stack(emissions_list), dim=0)
    avg_transitions = torch.mean(torch.stack(transitions_list), dim=0)

    # Create a temporary CRF layer for decoding
    num_tags = avg_emissions.size(-1)
    crf = CRF(num_tags, batch_first=True).to(avg_emissions.device)
    crf.transitions.data.copy_(avg_transitions)

    # Decode
    mask_bool = attention_masks.bool().to(avg_emissions.device)
    predicted_label_ids_batch = crf.decode(avg_emissions, mask=mask_bool)

    return predicted_label_ids_batch

if __name__ == "__main__":
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    model_checkpoint_path = [Path(MODEL_CHECKPOINT) for MODEL_CHECKPOINT in LIST_MODEL_CHECKPOINT]
    models = [load_model_from_checkpoint(path, device) for path in model_checkpoint_path]

    tokenizer= AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[TITLE]", "[ABSTRACT]"]})

    with open(PATH_EVAL_ARTICLES, "r") as f:
        data = json.load(f)
    df_infer = prepare_json_to_df(PATH_EVAL_ARTICLES, False, 'inference')

    inference_dataset_title = InferenceDataset(df_infer, tokenizer, MAX_SEQ_LEN, location="title")
    inference_dataset_abstract = InferenceDatasetCut(df_infer, tokenizer, MAX_SEQ_LEN, location="abstract", is_start=True)
    inference_dataset_abstract_extra = InferenceDatasetCut(df_infer, tokenizer, MAX_SEQ_LEN, location="abstract", is_start=False)
    
    inference_dataloader_title = DataLoader(
        inference_dataset_title,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_inference # Use custom collate_fn
    )

    inference_dataloader_abstract = DataLoader(
        inference_dataset_abstract,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_inference_extra # Use custom collate_fn
    )

    inference_dataloader_abstract_extra = DataLoader(
        inference_dataset_abstract_extra,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_inference_extra # Use custom collate_fn
    )

    results_title = {} # Dict to store final JSON output {pmid: {"entities": [...]}}
    print("Starting inference...")

    with torch.no_grad():
        for batch in tqdm(inference_dataloader_title, desc="Inference"):
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            offset_mappings = batch['offset_mappings'] # List of lists
            doc_ids = batch['doc_ids'] # List of doc IDs

            predicted_label_ids_batch = ensemble_model_decode(models, input_ids, attention_masks)

            for i in range(len(batch['input_ids'])):
                # print(f"DEBUG: Processing document ID: {doc_ids[i]}")
                result = decode_predictions(
                    predicted_label_ids_batch[i], 
                    tokenizer, input_ids[i], 
                    attention_masks[i], 
                    offset_mappings[i],
                    text=batch['title'][i], # Use title for inference
                    location="title"
                    ) # Decode predictions for this item
                
                results_title[doc_ids[i]] = {"entities": result} # Store results in the dictionary
        
        # for batch in tqdm(inference_dataloader_abstract, desc="Inference"):
        #     input_ids = batch['input_ids'].to(device)
        #     attention_masks = batch['attention_mask'].to(device)
        #     offset_mappings = batch['offset_mappings']
        #     doc_ids = batch['doc_ids']
        #     period_char_idx = batch['period_char_idx']

        #     predicted_label_ids_batch = ensemble_model_decode(models, input_ids, attention_masks)

        #     for i in range(len(batch['input_ids'])):
        #         # print(f"DEBUG: Processing document ID: {doc_ids[i]}")
        #         result = decode_predictions(
        #             predicted_label_ids_batch[i], 
        #             tokenizer, input_ids[i], 
        #             attention_masks[i], 
        #             offset_mappings[i],
        #             text=batch['abstract'][i], # Use abstract for inference
        #             location="abstract"
        #             )
                
        #         if doc_ids[i] in results_title:
        #             results_title['period_char_idx'] = period_char_idx
        #             results_title[doc_ids[i]]["entities"].extend(result)
                    
        #         else:
        #             results_title[doc_ids[i]] = {"entities": result}

        for batch in tqdm(inference_dataloader_abstract, desc="Inference"):
            print(batch.keys())
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            offset_mappings = batch['offset_mappings']
            doc_ids = batch['doc_ids']
            period_char_idxs = batch['period_char_idxs']

            predicted_label_ids_batch = ensemble_model_decode(models, input_ids, attention_masks)

            for i in range(len(batch['input_ids'])):
                # print(f"DEBUG: Processing document ID: {doc_ids[i]}")
                result = decode_predictions(
                    predicted_label_ids_batch[i], 
                    tokenizer, input_ids[i], 
                    attention_masks[i], 
                    offset_mappings[i],
                    text=batch['abstract'][i], # Use abstract for inference
                    location="abstract"
                    )
                
                if doc_ids[i] in results_title:
                    results_title[doc_ids[i]]['period_char_idx'] = period_char_idxs[i]
                    results_title[doc_ids[i]]["entities"].extend(result)
                    
                else:
                    results_title[doc_ids[i]]['period_char_idx'] = period_char_idxs[i]
                    results_title[doc_ids[i]] = {"entities": result}

        for batch in tqdm(inference_dataloader_abstract_extra, desc="Inference"):
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            offset_mappings = batch['offset_mappings']
            doc_ids = batch['doc_ids']
            period_char_idxs = batch['period_char_idxs']

            predicted_label_ids_batch = ensemble_model_decode(models, input_ids, attention_masks)

            for i in range(len(batch['input_ids'])):
                # print(f"DEBUG: Processing document ID: {doc_ids[i]}")
                period_char_idx = results_title[doc_ids[i]]['period_char_idx']
                if period_char_idx !=-1:
                    print(period_char_idx)
                    print(f"File {doc_ids[i]} abstract token length exceeds MAX_TOK_LEN, inference for extra part")
                    result = decode_predictions(
                        predicted_label_ids_batch[i], 
                        tokenizer, input_ids[i], 
                        attention_masks[i], 
                        offset_mappings[i],
                        text=batch['abstract'][i], # Use abstract for inference
                        location="abstract"
                        )
                    # for ent in result:
                    #     ent["start_idx"] += period_char_idx
                    #     ent["end_idx"] += period_char_idx
                    results_title[doc_ids[i]]["entities"].extend(result)
                    
        
    print("Inference completed.")
    
    output_path = "data/processed/inference_results_test.json"

    with open(output_path, "w") as f:
        json.dump(results_title, f, indent=4)