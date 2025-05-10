from typing import Any, Tuple
import json 
import os
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset

from transformers import (
    AutoModel,
    AutoTokenizer
)

from utils import (
    MAX_SEQ_LEN, 
    PATH_DEV_ANNOTATIONS, 
    DICT_PATH_TRAIN_ANNOTATIONS,
    BASE_MODEL_NAME, 
    PATH_LABEL_MAP,
    PROCESSED_DIR,)

# PATH_EVAL_ARTICLES = "data/raw/GutBrainIE_Full_Collection_2025/Articles/json_format/articles_dev.json"
PATH_AUG_ARTICLES = "notebooks\pubmed_gut_brain_500.json"

with open(PATH_LABEL_MAP, "r") as f:
    label_mappings = json.load(f)
    label2id = label_mappings['label2id']
    id2label = label_mappings['id2label']

def get_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    return tokenizer

def embed_text(tokenizer, text: str, entitie_labels):
    embed_output = tokenizer(
        text, 
        truncation=False,
        max_length=MAX_SEQ_LEN,
        return_offsets_mapping=True,
        padding=False,
        add_special_tokens=False
        )
    
    label_ids = embed_output['token_type_ids']

    if entitie_labels is None:
        return {
            'input_ids': embed_output['input_ids'],
            'attention_mask': embed_output['attention_mask'],
            'token_type_ids': None,
        }
    
    for entity in entitie_labels:
        start_idx, end_idx = entity['start_idx'], entity['end_idx']
        label = entity['label']
        if start_idx in [i[0] for i in embed_output['offset_mapping']] and end_idx+1 in [i[1] for i in embed_output['offset_mapping']]:
            offset_start_id = [i[0] for i in embed_output['offset_mapping']].index(start_idx)
            offset_end_id = [i[1] for i in embed_output['offset_mapping']].index(end_idx+1)
            label_ids[offset_start_id] = label2id[f'B-{label}']
            label_ids[offset_start_id+1:offset_end_id+1] = [label2id[f'I-{label}']] * (offset_end_id - offset_start_id)
        elif start_idx in [i[0] for i in embed_output['offset_mapping']] and end_idx+2 in [i[1] for i in embed_output['offset_mapping']]:
            print(f"实体标签错误: {entity}, 文本: {text}")
            print(f"start_idx: {start_idx}, end_idx: {end_idx}, offset_mapping: {embed_output['offset_mapping']}")  # TAMs TAM->TAMs IBS symptom->IBS symptoms
            offset_start_id = [i[0] for i in embed_output['offset_mapping']].index(start_idx)
            offset_end_id = [i[1] for i in embed_output['offset_mapping']].index(end_idx+2)
            label_ids[offset_start_id] = label2id[f'B-{label}']
            label_ids[offset_start_id+1:offset_end_id+1] = [label2id[f'I-{label}']] * (offset_end_id - offset_start_id)
        else:
            print(f"实体标签错误: {entity}, 文本: {text}")
            print(f"start_idx: {start_idx}, end_idx: {end_idx}, offset_mapping: {embed_output['offset_mapping']}") 

            raise NotImplementedError() # no need to handle this case if no error in training data
    return {
        'input_ids': embed_output['input_ids'],
        'attention_mask': embed_output['attention_mask'],
        'token_type_ids': label_ids,
    }

def embed_text_with_special_tokens(tokenizer, text, entities, location):
    vocab = tokenizer.get_vocab()

    text_embed_dict = embed_text(tokenizer=tokenizer, text=text, entitie_labels=entities)

    if location == 'abstract':
        special_token_id = vocab['[ABSTRACT]']
    elif location == 'title':
        special_token_id = vocab['[TITLE]']

    class_id = vocab['[CLS]']

    input_ids_title = [class_id] + [class_id] + text_embed_dict['input_ids']
    attention_mask = [1] + [1] + text_embed_dict['attention_mask']

    token_type_ids = [-100] + [0] + text_embed_dict['token_type_ids']
    assert len(input_ids_title) == len(attention_mask) == len(token_type_ids), f"Length mismatch: {len(input_ids_title)}, {len(attention_mask)}, {len(token_type_ids)}"

    if len(input_ids_title) > MAX_SEQ_LEN:
        input_ids_title = input_ids_title[:MAX_SEQ_LEN]
        attention_mask = attention_mask[:MAX_SEQ_LEN]
        token_type_ids = token_type_ids[:MAX_SEQ_LEN]
    elif len(input_ids_title) < MAX_SEQ_LEN:
        input_ids_title += [0] * (MAX_SEQ_LEN - len(input_ids_title))
        attention_mask += [0] * (MAX_SEQ_LEN - len(attention_mask))
        token_type_ids += [-100] * (MAX_SEQ_LEN - len(token_type_ids))

    assert len(input_ids_title) == len(attention_mask) == len(token_type_ids)==MAX_SEQ_LEN, f"Length mismatch: {len(input_ids_title)}, {len(attention_mask)}, {len(token_type_ids)}"

    return {
        'input_ids': input_ids_title,
        'attention_mask': attention_mask,
        'labels': token_type_ids,
    }



def batch_embed_text(tokenizer, texts: list, entities: list, locations: list) -> torch.Tensor:

    tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "locations": []}
    for text, entity, location in zip(texts, entities, locations):
        embed_output = embed_text_with_special_tokens(
            tokenizer=tokenizer, 
            text=text,
            entities=entity,
            location=location
        )
        input_ids, attention_mask, token_type_ids = embed_output['input_ids'], embed_output['attention_mask'], embed_output['labels']
        tokenized_inputs['input_ids'].append(input_ids)
        tokenized_inputs['attention_mask'].append(attention_mask)
        tokenized_inputs['labels'].append(token_type_ids)
        tokenized_inputs['locations'].append(location)

    return tokenized_inputs


def prepare_json_to_df(file_path: str, has_annotations, category) -> list:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if has_annotations:
        datas = []
        for id, content in data.items():
            datas.append({
                'id': id,
                'title': content['metadata']['title'],
                'abstract': content['metadata']['abstract'],
                'entities': content['entities'],
            })
    else:
        datas = []
        for id, content in data.items():
            datas.append({
            'id': id,
            'title': content['title'],
            'abstract': content['abstract'],
        })

    # Create DataFrame
    df = pd.DataFrame(datas)
    # df['category'] = category
    return df

def prepare_inference_dataset() -> list:
    tokenizer = get_tokenizer(BASE_MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[TITLE]", "[ABSTRACT]"]})

    df_infer = prepare_json_to_df(PATH_AUG_ARTICLES, has_annotations=False, category='dev')

    print(f"推理集: {df_infer.iloc[0]}")
    print(f"DataFrame形状: {df_infer.shape}")  # (行数, 列数)
    print(f"DataFrame是否为空: {df_infer.empty}")
    ds_infer = Dataset.from_dict({
    "id": df_infer["id"].tolist(),
    "title": df_infer["title"].tolist(),
    "abstract": df_infer["abstract"].tolist(),
})
    
    print(f"推理集大小: {len(ds_infer)}")
    # ds_infer = ds_infer.remove_columns(ds_infer.column_names)
    
    infer_output_path = os.path.join(PROCESSED_DIR, "infer_tokenized_dataset")

    print(f"保存推理集到 {infer_output_path}")
    # tokenized_infer.save_to_disk(infer_output_path)

class InferenceDataset(TorchDataset):
    def __init__(self, df_data, tokenizer, max_seq_length, location):
        self.data = df_data
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.location = location
        self.location_tok = "[ABSTRACT]" if location == 'abstract' else "[TITLE]"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        doc_id = item['id']
        title = item['title']
        abstract = item['abstract']

        if self.location == 'abstract':
            text = abstract
        elif self.location == 'title':
            text = title

        # Tokenize
        text_embed = self.tokenizer(
            text, 
            truncation=False,
            max_length=self.max_seq_length,
            return_offsets_mapping=True,
            padding=False,
            add_special_tokens=False
        )

        input_ids, attention_mask_title = text_embed['input_ids'], text_embed['attention_mask']
        offset_mapping = text_embed['offset_mapping']
        
        input_ids = [self.tokenizer.cls_token_id] + [self.tokenizer.convert_tokens_to_ids(self.location_tok)] + input_ids
        attention_mask = [1] + [1] + attention_mask_title

        # TODO: validate length
        # offset_mapping_abstract = [(start + len(input_ids_title) + 3, end + len(input_ids_title) + 3) for start, end in offset_mapping_abstract]
        offset_mapping = [(0, 0)] + offset_mapping + [(0, 0)]

        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            offset_mapping = offset_mapping[:self.max_seq_length]
        elif len(input_ids) < self.max_seq_length:
            input_ids += [0] * (self.max_seq_length - len(input_ids))
            attention_mask += [0] * (self.max_seq_length - len(attention_mask))
            offset_mapping += [(0, 0)] * (self.max_seq_length - len(offset_mapping))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        # offset_mapping = torch.Tensor(offset_mapping)

        return {
            "doc_id": doc_id,
            "title": title,
            "abstract": abstract,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping
        }


def prepare_dataset() -> list:
    tokenizer = get_tokenizer(BASE_MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[TITLE]", "[ABSTRACT]"]})

    df_dev = prepare_json_to_df(PATH_DEV_ANNOTATIONS, has_annotations=True, category='dev')
    df_dev['location'] = 'title'
    df_dev['title_entities'] = df_dev['entities'].apply(lambda x: [entity for entity in x if entity['location'] == 'title'])
    df_dev_title = df_dev[['id', 'title', 'title_entities', 'location']].copy().rename(columns={"title": "text", "title_entities": "entities"})
    df_dev['location'] = 'abstract'
    df_dev['abstract_entities'] = df_dev['entities'].apply(lambda x: [entity for entity in x if entity['location'] == 'abstract'])
    df_dev_abstract = df_dev[['id', 'abstract', 'abstract_entities', 'location']].copy().rename(columns={"abstract": "text", "abstract_entities": "entities"})
    df_dev = pd.concat([df_dev_title, df_dev_abstract], ignore_index=True)

    df_train = pd.concat([
        prepare_json_to_df(DICT_PATH_TRAIN_ANNOTATIONS['bronze'], has_annotations=True, category='bronze'),
        prepare_json_to_df(DICT_PATH_TRAIN_ANNOTATIONS['gold'], has_annotations=True, category='gold'),
        prepare_json_to_df(DICT_PATH_TRAIN_ANNOTATIONS['platinum'], has_annotations=True, category='platinum'),
        prepare_json_to_df(DICT_PATH_TRAIN_ANNOTATIONS['silver'], has_annotations=True, category='silver'),
        prepare_json_to_df(DICT_PATH_TRAIN_ANNOTATIONS['augment'], has_annotations=True, category='augment')
    ])
    df_train['location'] = 'title'
    df_train['title_entities'] = df_train['entities'].apply(lambda x: [entity for entity in x if entity['location'] == 'title'])
    df_train_title = df_train[['id', 'title', 'title_entities', 'location']].copy().rename(columns={"title": "text", "title_entities": "entities"})
    df_train['location'] = 'abstract'
    df_train['abstract_entities'] = df_train['entities'].apply(lambda x: [entity for entity in x if entity['location'] == 'abstract'])
    df_train_abstract = df_train[['id', 'abstract', 'abstract_entities', 'location']].copy().rename(columns={"abstract": "text", "abstract_entities": "entities"})
    df_train = pd.concat([df_train_title, df_train_abstract], ignore_index=True)

    ds_dev = Dataset.from_pandas(df_dev)
    ds_train = Dataset.from_pandas(df_train)

    # print(ds_dev['location'])

    tokenized_dev = ds_dev.map(
        lambda examples: batch_embed_text(tokenizer, examples['text'], examples['entities'], examples['location']),
        batched=True,
        remove_columns=ds_dev.column_names,
    )
    tokenized_train = ds_train.map(
        lambda examples: batch_embed_text(tokenizer, examples['text'], examples['entities'], examples['location']),
        batched=True,
        remove_columns=ds_train.column_names,
    )
    
    train_output_path = os.path.join(PROCESSED_DIR, "train_tokenized_dataset_seperate_aug")
    dev_output_path = os.path.join(PROCESSED_DIR, "dev_tokenized_dataset_seperate_aug")
    
    print(f"保存训练集到 {train_output_path}")
    tokenized_train.save_to_disk(train_output_path)
    
    print(f"保存验证集到 {dev_output_path}")
    tokenized_dev.save_to_disk(dev_output_path)

        # break
        # todo: >500 tokens, split into two sequences

    print(f"all embeded")
    print(f"train dataset size: {len(tokenized_train)}")
    print(f"dev dataset size: {len(tokenized_dev)}")

    return tokenized_train, tokenized_dev


def main():
    tokenized_train, tokenized_dev = prepare_dataset()
    print(f"train dataset size: {len(tokenized_train)}")
    print(f"dev dataset size: {len(tokenized_dev)}")
    
    print(f'embed train dataset: {tokenized_train[0]}')
    print(f'embed dev dataset: {tokenized_dev[0]}')
    # print(f'embed train dataset: {tokenized_train[0]}')
    # print(f'embed dev dataset: {tokenized_dev[0]}')

if __name__ == '__main__':
    main()