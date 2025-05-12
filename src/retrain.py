import yaml, os
import json
from pathlib import Path
import numpy as np
from torch.optim import AdamW
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_scheduler,
    set_seed,
)
from tqdm.auto import tqdm
from accelerate import Accelerator
from model_with_crf import BertCrfForTokenClassification

from utils import (
    BATCH_SIZE,
    SEED,
    NUM_TOKS,
    NUM_LABELS,
    BASE_MODEL_NAME,
    NUM_EPOCHS,
    WARMUP_STEPS,
    LR,
    WEIGHT_DECAY,
    MAX_SEQ_LEN,
    USE_MIXED_PRECISION,
    OUTPUT_DIR,
    compute_metrics
)
CKPT = 'models/biomedbert_finetuned/best_model7724'
from prepare_data import prepare_data_for_training

def load_config():
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir.parent / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        print(f"配置加载成功: {config_path}")
    return config

def load_model_from_checkpoint(model_checkpoint_path, num_labels, device):
    tokenizer = AutoTokenizer.from_pretrained(str(model_checkpoint_path))
    print(f"Loading model structure {BASE_MODEL_NAME} with {num_labels} labels...")
    model = BertCrfForTokenClassification(
        model_name_or_path=BASE_MODEL_NAME, # Use base name for initial structure
        num_labels=num_labels
    )
    model.resize_token_embeddings(len(tokenizer))

    print(f"Loading model weights from {model_checkpoint_path}...")
    base_model_state_dict = AutoModel.from_pretrained(str(model_checkpoint_path)).state_dict()
    model.bert.load_state_dict(base_model_state_dict, strict=False) # Use strict=False initially if needed
    print("Base BERT model weights loaded.")

    classifier_crf_state_path = os.path.join(model_checkpoint_path, "classifier_crf_state_dict.bin")
    custom_states = torch.load(classifier_crf_state_path, map_location='cpu')
    model.classifier.load_state_dict(custom_states['classifier_state_dict'])
    model.crf.load_state_dict(custom_states['crf_state_dict'])
    print("Classifier and CRF state dicts loaded.")

    model.to(device)
    print(f"Model loaded and moved to {device}.")
    return model

def get_tokenizer(model_name_or_path='data/processed/tokenizer'):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    return tokenizer


def get_model_and_config_from_ckpt(model_name_or_path, ckpt, num_labels):
    config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
    model = load_model_from_checkpoint(ckpt, num_labels, device='cuda')

    model.resize_token_embeddings(NUM_TOKS)
    return model, config

def get_optimizer_and_scheduler_seperate(model, max_train_steps):
    optimizer = AdamW(
    [{'params': model.bert.parameters(), 'lr': 2e-5},
    {'params': model.classifier.parameters(), 'lr': 2e-4},
    {'params': model.crf.parameters(), 'lr': 2e-3}  ]
    )

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=max_train_steps,
    )

    return optimizer, lr_scheduler

def get_optimizer_and_scheduler(model, max_train_steps):
    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=max_train_steps,
    )

    return optimizer, lr_scheduler

def get_training_args():
    training_args = {
        "model_name": BASE_MODEL_NAME,
        "num_train_epochs": NUM_EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": WARMUP_STEPS,
        "batch_size": BATCH_SIZE,
        "max_seq_length": MAX_SEQ_LEN ,
        "seed": SEED,
        "fp16": USE_MIXED_PRECISION,
    }
    with open(OUTPUT_DIR / "training_args.json", "w") as f:
        json.dump(training_args, f, indent=2)


def main():
    set_seed(SEED)
    train_dataloader, eval_dataloader, label_mappings, class_weights = prepare_data_for_training()
    tokenizer = get_tokenizer()
    accelerator = Accelerator(mixed_precision=USE_MIXED_PRECISION) # NEW LINE

    print(f"Device: {accelerator.device}")
    print(f"Mixed Precision: {accelerator.mixed_precision}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output dir: {output_dir}")

    model, config = get_model_and_config_from_ckpt(model_name_or_path=BASE_MODEL_NAME, ckpt=CKPT, num_labels=NUM_LABELS)
    model.to(accelerator.device)  # Move model to the correct device

    class_weights = class_weights.to(accelerator.device)  # Move class weights to the correct device

    max_train_steps = NUM_EPOCHS * len(train_dataloader)
    optimizer, lr_scheduler = get_optimizer_and_scheduler_seperate(model, max_train_steps)
    
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    progress_bar = tqdm(range(max_train_steps), desc="训练进度")
    best_f1 = 0
    patience = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_dataloader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                return_dict=True,
            )

            loss = outputs["loss"] #
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            progress_bar.update(1)

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - avg loss: {avg_epoch_loss:.4f}")

        model.eval()
        all_eval_labels = []
        all_logits = []
        all_masks = []
        eval_loss = 0.0
        for batch in tqdm(eval_dataloader, desc="evaluating"):
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch['labels'], # <--- 不传入 labels 进行预测
                    return_dict=True
                )
                loss = outputs["loss"]
                eval_loss += loss.item()

            
            # predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            logits = outputs["logits"]
            masks = batch["attention_mask"]

            all_logits.append(logits.cpu())
            all_masks.append(masks.cpu())
            all_eval_labels.append(labels.cpu())

            logits_cat_cpu = torch.cat(all_logits, dim=0)
            labels_cat_cpu = torch.cat(all_eval_labels, dim=0)
            masks_cat_cpu = torch.cat(all_masks, dim=0)

        print(f'eval loss: {eval_loss}')
        # print(all_masks[0][0])
        # print(all_eval_labels[0][0])

        model_device = accelerator.device 
        unwrapped_model = accelerator.unwrap_model(model)
        logits_cat_device = logits_cat_cpu.to(model_device)
        masks_cat_device = masks_cat_cpu.to(model_device)


        decoded_predictions = unwrapped_model.crf.decode(logits_cat_device, mask=masks_cat_device.bool())
        all_predictions_final = []
        all_labels_final = []

        for i, (pred_seq, label_seq, mask) in enumerate(zip(decoded_predictions, labels_cat_cpu.numpy().tolist(), masks_cat_device.cpu().numpy().tolist())):
            seq_len = sum(mask)

            valid_pred = pred_seq[:seq_len][1:] #！！！
            valid_label = [l for l in label_seq[:seq_len] if l != -100]
            
            valid_pred = valid_pred[:len(valid_label)]

            all_predictions_final.append(valid_pred)
            all_labels_final.append(valid_label)

        print(f"解码完成，预测序列数量: {len(all_predictions_final)}, 真实标签序列数量: {len(all_labels_final)}")
        if len(all_predictions_final) != len(all_labels_final):
                print(f"预测和标签数量不匹配! P: {len(all_predictions_final)}, L: {len(all_labels_final)}")

        metrics = {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0}
        if accelerator.is_main_process:
            if len(all_predictions_final) == len(all_labels_final) and len(all_labels_final) > 0: # 确保有数据且数量匹配
                try:
                    metrics = compute_metrics(all_predictions_final, all_labels_final, label_mappings)
                except Exception as e:
                    print(f"计算指标时出错: {e}", exc_info=True)
            else:
                print("预测或标签为空或数量不匹配，跳过指标计算。")

        
        print(f"Epoch {epoch+1} evaluation results:")
        print(f"  precision: {metrics['precision']:.4f}")
        print(f"  recall: {metrics['recall']:.4f}")
        print(f"  F1 score: {metrics['f1']:.4f}")

        if metrics['f1'] > best_f1:
            print(f"Eval loss decreased from {best_f1} to {eval_loss}")
            best_f1 = metrics['f1']
            patience = 0

            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint_dir = output_dir / "best_model"
            try:
                unwrapped_model.save_pretrained(checkpoint_dir)
                print("use save_pretrained succsefully saved")
            except Exception as e:
                print(f"save_pretrained failed: {e}. attempt to only save state_dict...")
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                accelerator.save(unwrapped_model.state_dict(), checkpoint_dir / "pytorch_model.bin")
                if hasattr(unwrapped_model, 'bert'):
                    unwrapped_model.bert.config.save_pretrained(checkpoint_dir)
                else:
                    print("cannot auto save config")

            tokenizer.save_pretrained(checkpoint_dir)
        else:
            patience += 1
            print(f"No improvement, current patience: {patience}")
            if patience >= 5:
                print("Early stop")
                break
        if patience >= 5:
            print("Early stop")
            break

        final_model_dir = output_dir / "final_model"
        unwrapped_model = accelerator.unwrap_model(model)
        try:
             unwrapped_model.save_pretrained(final_model_dir)
        except Exception as e:
             print(f"save_pretrained failed: {e}. attempt to save only state_dict...")
             final_model_dir.mkdir(parents=True, exist_ok=True)
             accelerator.save(unwrapped_model.state_dict(), final_model_dir / "pytorch_model.bin")
             if hasattr(unwrapped_model, 'bert'):
                  unwrapped_model.bert.config.save_pretrained(final_model_dir)

        tokenizer.save_pretrained(final_model_dir)
        print(f"Final model saved to {final_model_dir}")
    

if __name__ == "__main__":
    main()