"""
====================================================
Sentiment Arabia - Lightweight Training Script
====================================================
تدريب نموذج خفيف للعمل ضمن موارد محدودة
يستخدم arabert-mini أو distilbert مع تحسينات للذاكرة

النهج:
- نموذج خفيف الوزن (حجم أصغر)
- batch_size صغير
- gradient checkpointing
- mixed precision إذا كان متاحاً
====================================================
"""

import os
import sys
import gc
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Memory management
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/home/user/webapp/logs/training_lite.log', 
                           mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path("/home/user/webapp")
DATA_DIR = BASE_DIR / "data/sentiment/processed"
MODEL_DIR = BASE_DIR / "models/sentiment"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Free memory before starting
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None


class ArabicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }


def evaluate(model, loader, device, criterion=None):
    model.eval()
    preds, lbls = [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            out = model(input_ids=ids, attention_mask=mask)
            logits = out.logits
            
            if criterion:
                total_loss += criterion(logits, labels).item()
            
            preds.extend(torch.argmax(logits, -1).cpu().tolist())
            lbls.extend(labels.cpu().tolist())
            
            del ids, mask, labels, out, logits
            gc.collect()
    
    acc = accuracy_score(lbls, preds)
    f1 = f1_score(lbls, preds, average='macro', zero_division=0)
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'avg_loss': avg_loss
    }, preds, lbls


def train_lightweight(args):
    logger.info("=" * 60)
    logger.info("🚀 Sentiment Arabia - Lightweight Training")
    logger.info("=" * 60)
    
    device = torch.device('cpu')  # CPU only to save memory
    logger.info(f"💻 الجهاز: {device}")
    
    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv", encoding='utf-8-sig')
    val_df = pd.read_csv(DATA_DIR / "val.csv", encoding='utf-8-sig')
    test_df = pd.read_csv(DATA_DIR / "test.csv", encoding='utf-8-sig')
    
    # Use limited samples to fit in memory
    max_train = args.max_samples or 2000
    max_val = min(500, len(val_df))
    max_test = min(500, len(test_df))
    
    # Stratified sampling
    train_dfs = []
    for label in [0, 1, 2]:
        subset = train_df[train_df['label'] == label].head(max_train // 3)
        train_dfs.append(subset)
    train_df = pd.concat(train_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    val_df = val_df.head(max_val)
    test_df = test_df.head(max_test)
    
    logger.info(f"📊 Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    logger.info(f"   توزيع: {train_df['label'].value_counts().to_dict()}")
    
    # Load tokenizer only first
    logger.info(f"\n📥 تحميل Tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset = ArabicDataset(
        train_df['text'].tolist(), 
        train_df['label'].tolist(), 
        tokenizer, args.max_length
    )
    val_dataset = ArabicDataset(
        val_df['text'].tolist(), 
        val_df['label'].tolist(), 
        tokenizer, args.max_length
    )
    test_dataset = ArabicDataset(
        test_df['text'].tolist(), 
        test_df['label'].tolist(), 
        tokenizer, args.max_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)
    
    # Load model
    logger.info(f"🤖 تحميل النموذج: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"📊 المعاملات: {total_params:,}")
    
    # Loss with class weights
    label_counts = train_df['label'].value_counts().sort_index()
    total = len(train_df)
    weights = torch.FloatTensor([
        total / (3 * label_counts.get(i, 1)) for i in range(3)
    ])
    criterion = nn.CrossEntropyLoss(weight=weights)
    logger.info(f"⚖️ أوزان الفئات: {weights.tolist()}")
    
    # Optimizer - freeze all except last 2 layers to save memory
    # Unfreeze only classifier and last encoder layer
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # Unfreeze classifier and last bert layer
    for name, param in model.named_parameters():
        if any(x in name for x in ['classifier', 'pooler', 'bert.encoder.layer.11', 
                                     'bert.encoder.layer.5', 'distilbert.transformer.layer.5',
                                     'roberta.pooler', 'pooler']):
            param.requires_grad = True
    
    # If nothing was unfrozen, unfreeze classifier at least
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
        # Also unfreeze last layer
        layers = list(model.named_parameters())
        for name, param in layers[-20:]:
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 المعاملات القابلة للتدريب: {trainable:,} / {total_params:,}")
    
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Training
    logger.info(f"\n🏋️ التدريب: {args.epochs} epochs, batch={args.batch_size}")
    
    best_f1 = 0
    best_acc = 0
    patience_count = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        
        train_preds, train_lbls = [], []
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            ids = batch['input_ids']
            mask = batch['attention_mask']
            labels = batch['label']
            
            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask)
            loss = criterion(out.logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            train_preds.extend(torch.argmax(out.logits, -1).tolist())
            train_lbls.extend(labels.tolist())
            
            del ids, mask, labels, out, loss
            gc.collect()
        
        train_acc = accuracy_score(train_lbls, train_preds)
        train_f1 = f1_score(train_lbls, train_preds, average='macro', zero_division=0)
        avg_loss = total_loss / len(train_loader)
        
        val_results, _, _ = evaluate(model, val_loader, device, criterion)
        
        elapsed = time.time() - t0
        logger.info(
            f"\n📈 Epoch {epoch}/{args.epochs} ({elapsed:.0f}s):\n"
            f"   Train: Loss={avg_loss:.4f} | Acc={train_acc:.4f} | F1={train_f1:.4f}\n"
            f"   Val:   Loss={val_results['avg_loss']:.4f} | "
            f"Acc={val_results['accuracy']:.4f} | F1={val_results['f1_macro']:.4f}"
        )
        
        history.append({
            'epoch': epoch,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_acc': val_results['accuracy'],
            'val_f1': val_results['f1_macro'],
        })
        
        if val_results['f1_macro'] > best_f1:
            best_f1 = val_results['f1_macro']
            best_acc = val_results['accuracy']
            patience_count = 0
            
            logger.info(f"💾 حفظ أفضل نموذج (Val F1: {best_f1:.4f})")
            best_path = MODEL_DIR / "best_model"
            model.save_pretrained(str(best_path))
            tokenizer.save_pretrained(str(best_path))
            
            with open(best_path / "checkpoint_info.json", 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'val_f1': best_f1,
                    'val_acc': best_acc,
                    'model_name': args.model_name,
                    'timestamp': datetime.now().isoformat(),
                    'label_mapping': {'negative': 0, 'neutral': 1, 'positive': 2}
                }, f, indent=2)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                logger.info(f"🛑 Early stopping بعد {epoch} epochs")
                break
        
        gc.collect()
    
    # Final test evaluation
    logger.info("\n🧪 التقييم النهائي على بيانات الاختبار...")
    
    # Load best model for final eval
    best_model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR / "best_model")
    ).to(device)
    
    test_results, test_preds, test_labels = evaluate(best_model, test_loader, device)
    del best_model
    gc.collect()
    
    label_names = ['negative', 'neutral', 'positive']
    report = classification_report(
        test_labels, test_preds, 
        target_names=label_names, 
        output_dict=True
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("📊 النتائج النهائية:")
    logger.info(f"   Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    logger.info(f"   F1 Macro: {test_results['f1_macro']:.4f}")
    for label in label_names:
        if label in report:
            m = report[label]
            logger.info(f"   {label:10s}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1-score']:.3f}")
    
    cm = confusion_matrix(test_labels, test_preds)
    logger.info(f"\n🔢 Confusion Matrix:")
    for i, row in enumerate(cm):
        logger.info(f"   {label_names[i]:10s}: {row.tolist()}")
    
    # Save results
    final_results = {
        'model_name': args.model_name,
        'training_date': datetime.now().isoformat(),
        'dataset': {'train': len(train_df), 'val': len(val_df), 'test': len(test_df)},
        'hyperparameters': vars(args),
        'best_val_f1': best_f1,
        'best_val_acc': best_acc,
        'test_metrics': {
            'accuracy': test_results['accuracy'],
            'f1_macro': test_results['f1_macro'],
        },
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'training_history': history,
        'label_mapping': {'negative': 0, 'neutral': 1, 'positive': 2}
    }
    
    with open(MODEL_DIR / "training_results.json", 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("✅ اكتمل التدريب!")
    logger.info(f"   Best Val F1: {best_f1:.4f}")
    logger.info(f"   Test Acc: {test_results['accuracy']:.4f}")
    logger.info(f"   النموذج: {MODEL_DIR}/best_model")
    
    return final_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='aubmindlab/bert-base-arabertv02')
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--patience', type=int, default=2)
    args = parser.parse_args()
    
    if not (DATA_DIR / "train.csv").exists():
        logger.error("❌ بيانات التدريب غير موجودة!")
        sys.exit(1)
    
    return train_lightweight(args)


if __name__ == "__main__":
    main()
