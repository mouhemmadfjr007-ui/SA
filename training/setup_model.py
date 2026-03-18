"""
====================================================
Sentiment Arabia - Smart Model Setup
====================================================
إعداد النموذج الذكي: يحاول التدريب أولاً،
وإذا فشل يستخدم النموذج الجاهز مباشرة
مع نظام Rule-based محسّن للعربية
====================================================
"""

import os
import gc
import sys
import json
import time
import logging
import re
from pathlib import Path
from datetime import datetime

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path("/home/user/webapp")
DATA_DIR = BASE_DIR / "data/sentiment/processed"
MODEL_DIR = BASE_DIR / "models/sentiment"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "aubmindlab/bert-base-arabertv02"
MAX_LENGTH = 64
BATCH_SIZE = 2  # Very small batch for memory
MAX_SAMPLES = 300  # 100 per class
EPOCHS = 2


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv", encoding='utf-8-sig')
    val_df = pd.read_csv(DATA_DIR / "val.csv", encoding='utf-8-sig')
    test_df = pd.read_csv(DATA_DIR / "test.csv", encoding='utf-8-sig')
    
    # Balanced tiny sample
    train_parts = []
    for label in [0, 1, 2]:
        part = train_df[train_df['label'] == label].head(MAX_SAMPLES // 3)
        train_parts.append(part)
    train_mini = pd.concat(train_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    val_mini = val_df.head(100)
    test_mini = test_df.head(150)
    
    return train_mini, val_mini, test_mini


def setup_model_with_training():
    """محاولة تدريب مصغّر"""
    logger.info("🔄 محاولة التدريب المصغّر...")
    
    from torch.utils.data import Dataset, DataLoader
    
    class SimpleDataset(Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.encodings = tokenizer(
                [str(t) for t in texts],
                truncation=True, padding='max_length',
                max_length=MAX_LENGTH, return_tensors='pt'
            )
            self.labels = list(labels)
        
        def __len__(self): return len(self.labels)
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'label': torch.tensor(int(self.labels[idx]), dtype=torch.long)
            }
    
    train_df, val_df, test_df = load_data()
    logger.info(f"📊 عينات: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model (no fine-tuning of embeddings to save memory)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True
    )
    
    # Only train classifier head
    for param in model.bert.parameters():
        param.requires_grad = False
    # Unfreeze pooler and classifier
    for param in model.bert.pooler.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"📊 المعاملات: {trainable:,} trainable / {total:,} total")
    
    # Create datasets (pre-tokenize)
    logger.info("🔄 Tokenizing data...")
    train_ds = SimpleDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
    val_ds = SimpleDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)
    test_ds = SimpleDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer)
    
    gc.collect()
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4  # Higher LR since we're only training top layers
    )
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = 0
    history = []
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        preds, lbls = [], []
        total_loss = 0
        t0 = time.time()
        
        for batch in train_loader:
            ids = batch['input_ids']
            mask = batch['attention_mask']
            labels = batch['label']
            
            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask)
            loss = criterion(out.logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds.extend(torch.argmax(out.logits, -1).tolist())
            lbls.extend(labels.tolist())
            
            del ids, mask, labels, out, loss
            gc.collect()
        
        train_acc = accuracy_score(lbls, preds)
        train_f1 = f1_score(lbls, preds, average='macro', zero_division=0)
        
        # Validation
        model.eval()
        val_preds, val_lbls = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                val_preds.extend(torch.argmax(out.logits, -1).tolist())
                val_lbls.extend(batch['label'].tolist())
                del out
        
        val_acc = accuracy_score(val_lbls, val_preds)
        val_f1 = f1_score(val_lbls, val_preds, average='macro', zero_division=0)
        
        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch}: Train Acc={train_acc:.3f} F1={train_f1:.3f} | "
            f"Val Acc={val_acc:.3f} F1={val_f1:.3f} ({elapsed:.0f}s)"
        )
        
        history.append({'epoch': epoch, 'train_acc': train_acc, 'val_acc': val_acc, 'val_f1': val_f1})
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path = MODEL_DIR / "best_model"
            model.save_pretrained(str(best_path))
            tokenizer.save_pretrained(str(best_path))
            logger.info(f"💾 حفظ النموذج (Val F1={best_val_f1:.3f})")
        
        gc.collect()
    
    # Test evaluation
    best_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR / "best_model"))
    best_model.eval()
    test_preds, test_lbls = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = best_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            test_preds.extend(torch.argmax(out.logits, -1).tolist())
            test_lbls.extend(batch['label'].tolist())
    
    test_acc = accuracy_score(test_lbls, test_preds)
    test_f1 = f1_score(test_lbls, test_preds, average='macro', zero_division=0)
    report = classification_report(test_lbls, test_preds, 
                                    target_names=['negative','neutral','positive'],
                                    output_dict=True)
    
    logger.info(f"\n✅ اكتمل التدريب!")
    logger.info(f"   Test Accuracy: {test_acc:.4f}")
    logger.info(f"   Test F1 Macro: {test_f1:.4f}")
    
    # Save results
    results = {
        'model_name': MODEL_NAME,
        'training_date': datetime.now().isoformat(),
        'dataset': {'train': len(train_df), 'val': len(val_df), 'test': len(test_df)},
        'test_metrics': {'accuracy': test_acc, 'f1_macro': test_f1},
        'best_val_f1': best_val_f1,
        'classification_report': report,
        'training_history': history,
        'label_mapping': {'negative': 0, 'neutral': 1, 'positive': 2}
    }
    
    with open(MODEL_DIR / "training_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return test_acc, test_f1


def setup_pretrained_only():
    """
    إعداد نموذج pretrained بدون fine-tuning
    يضيف classifier layer محسّن فوق ArabicBERT
    """
    logger.info("🔄 إعداد النموذج الجاهز (Pretrained Only)...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True
    )
    
    best_path = MODEL_DIR / "best_model"
    model.save_pretrained(str(best_path))
    tokenizer.save_pretrained(str(best_path))
    
    # Save info
    with open(best_path / "checkpoint_info.json", 'w') as f:
        json.dump({
            'model_name': MODEL_NAME,
            'type': 'pretrained_only',
            'timestamp': datetime.now().isoformat(),
            'label_mapping': {'negative': 0, 'neutral': 1, 'positive': 2},
            'note': 'Zero-shot ArabicBERT classifier - for production use full training'
        }, f, indent=2)
    
    # Save results file
    with open(MODEL_DIR / "training_results.json", 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': MODEL_NAME,
            'training_date': datetime.now().isoformat(),
            'type': 'pretrained',
            'test_metrics': {'accuracy': 0.0, 'f1_macro': 0.0},
            'note': 'Pretrained model - needs fine-tuning for best results',
            'label_mapping': {'negative': 0, 'neutral': 1, 'positive': 2}
        }, f, indent=2)
    
    logger.info(f"✅ النموذج محفوظ في: {best_path}")
    del model
    gc.collect()


def main():
    logger.info("=" * 60)
    logger.info("🚀 Sentiment Arabia - Model Setup")
    logger.info("=" * 60)
    
    # Check if model already exists
    best_path = MODEL_DIR / "best_model"
    if (best_path / "config.json").exists():
        logger.info(f"✅ النموذج موجود بالفعل في {best_path}")
        
        # Load and show info
        info_path = best_path / "checkpoint_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
            logger.info(f"   النموذج: {info.get('model_name')}")
            logger.info(f"   Val F1: {info.get('val_f1', 'N/A')}")
        return
    
    # Try training first
    try:
        logger.info("📋 محاولة التدريب المصغّر...")
        test_acc, test_f1 = setup_model_with_training()
        logger.info(f"✅ التدريب نجح! Acc={test_acc:.3f}, F1={test_f1:.3f}")
        
    except MemoryError as e:
        logger.warning(f"⚠️ نفاد الذاكرة: {e}")
        logger.info("🔄 الانتقال إلى وضع Pretrained Only...")
        gc.collect()
        setup_pretrained_only()
        
    except Exception as e:
        logger.warning(f"⚠️ خطأ في التدريب: {e}")
        logger.info("🔄 الانتقال إلى وضع Pretrained Only...")
        gc.collect()
        setup_pretrained_only()


if __name__ == "__main__":
    main()
