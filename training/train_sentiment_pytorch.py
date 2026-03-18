"""
====================================================
Sentiment Arabia - PyTorch Training Script
====================================================
تدريب نموذج ArabicBERT لتحليل المشاعر
بدون Trainer - حلقة تدريب يدوية مستقرة

النموذج: aubmindlab/bert-base-arabertv02
المهمة: تصنيف ثلاثي (negative=0, neutral=1, positive=2)
====================================================
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/home/user/webapp/logs/training.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path("/home/user/webapp")
DATA_DIR = BASE_DIR / "data/sentiment/processed"
MODEL_DIR = BASE_DIR / "models/sentiment"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class ArabicSentimentDataset(Dataset):
    """
    Dataset مخصص لبيانات المشاعر العربية
    """
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts = df['text'].astype(str).tolist()
        self.labels = df['label'].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding.get('token_type_ids', 
                torch.zeros(self.max_length, dtype=torch.long)).squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def clean_arabic_text(text: str) -> str:
    """تنظيف بسيط للنص العربي"""
    import re
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_data(data_dir: Path, max_samples: int = None) -> tuple:
    """
    تحميل بيانات التدريب والتحقق والاختبار
    """
    train_df = pd.read_csv(data_dir / "train.csv", encoding='utf-8-sig')
    val_df = pd.read_csv(data_dir / "val.csv", encoding='utf-8-sig')
    test_df = pd.read_csv(data_dir / "test.csv", encoding='utf-8-sig')
    
    # Clean texts
    for df in [train_df, val_df, test_df]:
        df['text'] = df['text'].apply(clean_arabic_text)
        df.dropna(subset=['text', 'label'], inplace=True)
        df = df[df['text'].str.len() > 0]
    
    # Limit samples for testing
    if max_samples:
        train_df = train_df.head(max_samples)
        val_df = val_df.head(max_samples // 4)
        test_df = test_df.head(max_samples // 4)
    
    logger.info(f"📊 البيانات المحملة:")
    logger.info(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    logger.info(f"   توزيع التدريب: {train_df['label'].value_counts().to_dict()}")
    
    return train_df, val_df, test_df


def calculate_class_weights(train_df: pd.DataFrame, num_classes: int = 3) -> torch.Tensor:
    """
    حساب أوزان الفئات لمعالجة عدم التوازن
    """
    label_counts = train_df['label'].value_counts().sort_index()
    total = len(train_df)
    weights = []
    
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        weight = total / (num_classes * count)
        weights.append(weight)
    
    logger.info(f"⚖️ أوزان الفئات: {[f'{w:.3f}' for w in weights]}")
    return torch.FloatTensor(weights)


def evaluate_model(model, dataloader, device, criterion=None) -> dict:
    """
    تقييم النموذج على مجموعة بيانات
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            logits = outputs.logits
            
            if criterion:
                loss = criterion(logits, labels)
                total_loss += loss.item()
                num_batches += 1
            
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'avg_loss': total_loss / num_batches if num_batches > 0 else 0
    }
    
    return results, all_preds, all_labels


def train_model(args):
    """
    الدالة الرئيسية للتدريب
    """
    logger.info("=" * 60)
    logger.info("🚀 Sentiment Arabia - بدء التدريب")
    logger.info("=" * 60)
    logger.info(f"⚙️ الإعدادات: {vars(args)}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"💻 الجهاز: {device}")
    
    # Load data
    train_df, val_df, test_df = load_data(DATA_DIR, args.max_samples)
    
    # Load tokenizer
    logger.info(f"\n📥 تحميل Tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset = ArabicSentimentDataset(train_df, tokenizer, args.max_length)
    val_dataset = ArabicSentimentDataset(val_df, tokenizer, args.max_length)
    test_dataset = ArabicSentimentDataset(test_df, tokenizer, args.max_length)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size * 2, 
        shuffle=False, 
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size * 2, 
        shuffle=False, 
        num_workers=0
    )
    
    # Load model
    logger.info(f"\n🤖 تحميل النموذج: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 إجمالي المعاملات: {total_params:,}")
    logger.info(f"📊 المعاملات القابلة للتدريب: {trainable_params:,}")
    
    # Class weights
    class_weights = calculate_class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer - different LR for different layers
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=0.1  # 10% warmup
    )
    
    # Training loop
    logger.info(f"\n🏋️ بدء التدريب: {args.epochs} epoch(s), {len(train_loader)} batch/epoch")
    
    best_val_f1 = 0
    best_val_acc = 0
    training_history = []
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            
            # Log progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                current_loss = train_loss / (batch_idx + 1)
                current_acc = accuracy_score(train_labels, train_preds)
                logger.info(
                    f"  Epoch {epoch}/{args.epochs} | "
                    f"Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Loss: {current_loss:.4f} | "
                    f"Acc: {current_acc:.4f}"
                )
        
        # Train metrics
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        val_results, val_preds, val_labels = evaluate_model(model, val_loader, device, criterion)
        
        epoch_time = time.time() - epoch_start
        
        # Log epoch results
        logger.info(
            f"\n📈 Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s):\n"
            f"   Train - Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}\n"
            f"   Val   - Loss: {val_results['avg_loss']:.4f} | Acc: {val_results['accuracy']:.4f} | "
            f"F1: {val_results['f1_macro']:.4f}"
        )
        
        # Save history
        epoch_stats = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_results['avg_loss'],
            'val_acc': val_results['accuracy'],
            'val_f1': val_results['f1_macro'],
        }
        training_history.append(epoch_stats)
        
        # Save best model
        if val_results['f1_macro'] > best_val_f1:
            best_val_f1 = val_results['f1_macro']
            best_val_acc = val_results['accuracy']
            patience_counter = 0
            
            logger.info(f"💾 حفظ أفضل نموذج (F1: {best_val_f1:.4f})")
            model.save_pretrained(str(MODEL_DIR / "best_model"))
            tokenizer.save_pretrained(str(MODEL_DIR / "best_model"))
            
            # Save checkpoint info
            checkpoint_info = {
                'epoch': epoch,
                'val_f1': best_val_f1,
                'val_acc': best_val_acc,
                'model_name': args.model_name,
                'timestamp': datetime.now().isoformat()
            }
            with open(MODEL_DIR / "best_model" / "checkpoint_info.json", 'w') as f:
                json.dump(checkpoint_info, f, indent=2)
        else:
            patience_counter += 1
            logger.info(f"⏳ Patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"🛑 Early stopping بعد {epoch} epochs")
            break
    
    # Final evaluation on test set
    logger.info("\n🧪 تقييم النهائي على مجموعة الاختبار...")
    
    # Load best model
    best_model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR / "best_model")
    ).to(device)
    
    test_results, test_preds, test_labels = evaluate_model(best_model, test_loader, device)
    
    # Detailed report
    label_names = ['negative', 'neutral', 'positive']
    report = classification_report(
        test_labels, test_preds, 
        target_names=label_names,
        output_dict=True
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("📊 نتائج التقييم النهائي:")
    logger.info(f"{'='*60}")
    logger.info(f"   Accuracy:  {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    logger.info(f"   F1 Macro:  {test_results['f1_macro']:.4f}")
    logger.info(f"   F1 Weighted: {test_results['f1_weighted']:.4f}")
    logger.info(f"   Precision: {test_results['precision']:.4f}")
    logger.info(f"   Recall:    {test_results['recall']:.4f}")
    logger.info(f"\n📋 تقرير مفصل:")
    for label in label_names:
        if label in report:
            metrics = report[label]
            logger.info(
                f"   {label:10s}: P={metrics['precision']:.3f} | "
                f"R={metrics['recall']:.3f} | "
                f"F1={metrics['f1-score']:.3f} | "
                f"Support={int(metrics['support'])}"
            )
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    logger.info(f"\n🔢 Confusion Matrix:")
    logger.info(f"   Labels: {label_names}")
    for i, row in enumerate(cm):
        logger.info(f"   {label_names[i]:10s}: {row}")
    
    # Save all results
    final_results = {
        'model_name': args.model_name,
        'training_date': datetime.now().isoformat(),
        'dataset_stats': {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
        },
        'hyperparameters': vars(args),
        'best_val_metrics': {
            'f1_macro': best_val_f1,
            'accuracy': best_val_acc
        },
        'test_metrics': test_results,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'training_history': training_history,
        'label_mapping': {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }
    }
    
    results_path = MODEL_DIR / "training_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n💾 النتائج محفوظة في: {results_path}")
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ اكتمل التدريب!")
    logger.info(f"   أفضل F1 (validation): {best_val_f1:.4f}")
    logger.info(f"   Test Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"   النموذج محفوظ في: {MODEL_DIR}/best_model")
    logger.info(f"{'='*60}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Sentiment Arabia - Model Training')
    
    # Model settings
    parser.add_argument('--model_name', type=str, 
                       default='aubmindlab/bert-base-arabertv02',
                       help='اسم نموذج HuggingFace')
    
    # Data settings
    parser.add_argument('--max_length', type=int, default=128,
                       help='الحد الأقصى لطول التسلسل')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='الحد الأقصى لعدد العينات (للاختبار السريع)')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=5,
                       help='عدد epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='حجم الدفعة')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='معدل التعلم')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='معامل التراجع')
    parser.add_argument('--patience', type=int, default=3,
                       help='صبر Early Stopping')
    
    args = parser.parse_args()
    
    # Validate data exists
    if not (DATA_DIR / "train.csv").exists():
        logger.error("❌ لم يتم العثور على بيانات التدريب!")
        logger.error(f"   تأكد من تشغيل: python3 scripts/download_datasets.py أولاً")
        sys.exit(1)
    
    # Start training
    results = train_model(args)
    
    return results


if __name__ == "__main__":
    main()
