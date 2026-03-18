"""
Sentiment Arabia - Final Training Script
تدريب نموذج ArabicBERT على تصنيف المشاعر مع حفظ الأوزان
"""
import os, sys, time, json, logging, argparse
import numpy as np
import pandas as pd
from pathlib import Path

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '2'

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

# ─── Args ────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--max_samples', type=int, default=2000)
    p.add_argument('--epochs',      type=int, default=3)
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--lr',          type=float, default=2e-5)
    p.add_argument('--max_length',  type=int, default=64)
    p.add_argument('--patience',    type=int, default=2)
    return p.parse_args()

# ─── Dataset ─────────────────────────────────────────────────────────
class ArabicSentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=64):
        self.texts  = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tok    = tokenizer
        self.max_len = max_length

    def __len__(self):  return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ─── Main ─────────────────────────────────────────────────────────────
def main():
    args = get_args()
    DATA_DIR  = Path('/home/user/webapp/data/sentiment/processed')
    MODEL_OUT = Path('/home/user/webapp/models/sentiment/best_model')
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    logger.info(f"⚙️  max_samples={args.max_samples}  epochs={args.epochs}  bs={args.batch_size}  lr={args.lr}")

    # ─── Load Data ───────────────────────────────────────────────────
    def load_df(name, n=None):
        path = DATA_DIR / name
        if not path.exists():
            logger.error(f"❌ File not found: {path}"); sys.exit(1)
        df = pd.read_csv(path)
        # ensure label column
        if 'label' not in df.columns:
            lmap = {'negative': 0, 'neutral': 1, 'positive': 2}
            df['label'] = df['sentiment'].map(lmap)
        df = df.dropna(subset=['text', 'label'])
        df['label'] = df['label'].astype(int)
        if n: df = df.sample(min(n, len(df)), random_state=42)
        return df.reset_index(drop=True)

    train_df = load_df('train.csv', args.max_samples)
    val_df   = load_df('val.csv',   min(500, args.max_samples // 2))
    test_df  = load_df('test.csv',  min(500, args.max_samples // 2))

    logger.info(f"📊 Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")
    logger.info(f"   Label dist: {dict(train_df['label'].value_counts().sort_index())}")

    # ─── Tokenizer + Model ───────────────────────────────────────────
    MODEL_NAME = 'aubmindlab/bert-base-arabertv02'
    logger.info(f"⬇️  Loading tokenizer/model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"💻 Device: {device}")
    model.to(device)

    # ─── DataLoaders ─────────────────────────────────────────────────
    train_ds = ArabicSentimentDataset(train_df, tokenizer, args.max_length)
    val_ds   = ArabicSentimentDataset(val_df,   tokenizer, args.max_length)
    test_ds  = ArabicSentimentDataset(test_df,  tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=0)

    # ─── Class Weights ───────────────────────────────────────────────
    counts = np.bincount(train_df['label'].values, minlength=3).astype(float)
    weights = torch.tensor(counts.sum() / (3 * counts + 1e-8), dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    logger.info(f"⚖️  Class weights: {weights.tolist()}")

    # ─── Optimizer + Scheduler ───────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    # ─── Eval Helper ─────────────────────────────────────────────────
    def evaluate(loader):
        model.eval()
        all_preds, all_labels, total_loss = [], [], 0.0
        with torch.no_grad():
            for batch in loader:
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                lbls = batch['labels'].to(device)
                out  = model(input_ids=ids, attention_mask=mask)
                total_loss += loss_fn(out.logits, lbls).item()
                preds = out.logits.argmax(-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(lbls.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        return acc, f1, total_loss / len(loader)

    # ─── Training Loop ───────────────────────────────────────────────
    best_f1 = 0.0
    patience_cnt = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for i, batch in enumerate(train_loader):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['labels'].to(device)

            optimizer.zero_grad()
            out  = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

            if (i + 1) % 20 == 0:
                logger.info(f"  Epoch {epoch} step {i+1}/{len(train_loader)}  loss={loss.item():.4f}")

        val_acc, val_f1, val_loss = evaluate(val_loader)
        elapsed = time.time() - t0
        logger.info(f"✅ Epoch {epoch}/{args.epochs}  "
                    f"train_loss={train_loss/len(train_loader):.4f}  "
                    f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  "
                    f"time={elapsed:.1f}s")
        history.append({'epoch': epoch, 'val_acc': val_acc, 'val_f1': val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_cnt = 0
            # ─── Save model ──────────────────────────────────────────
            model.save_pretrained(str(MODEL_OUT))
            tokenizer.save_pretrained(str(MODEL_OUT))

            meta = {
                'epoch': epoch, 'val_acc': float(val_acc), 'val_f1': float(best_f1),
                'model_name': MODEL_NAME, 'max_length': args.max_length,
                'labels': {'0': 'negative', '1': 'neutral', '2': 'positive'},
                'label2id': {'negative': 0, 'neutral': 1, 'positive': 2},
                'id2label':  {'0': 'negative', '1': 'neutral', '2': 'positive'},
                'trained_samples': len(train_df),
                'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(MODEL_OUT / 'checkpoint_info.json', 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Saved best model  F1={best_f1:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info(f"⏹  Early stopping at epoch {epoch}")
                break

    # ─── Final Test Evaluation ───────────────────────────────────────
    test_acc, test_f1, _ = evaluate(test_loader)
    logger.info(f"\n{'='*50}")
    logger.info(f"📊 FINAL TEST RESULTS:")
    logger.info(f"   Accuracy = {test_acc:.4f} ({test_acc*100:.1f}%)")
    logger.info(f"   F1 Macro = {test_f1:.4f}")
    logger.info(f"   Best Val F1 = {best_f1:.4f}")
    logger.info(f"{'='*50}")

    # ─── Classification Report ───────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            all_preds.extend(out.logits.argmax(-1).cpu().numpy())
            all_labels.extend(batch['labels'].numpy())

    print(classification_report(all_labels, all_preds,
                                 target_names=['negative', 'neutral', 'positive'],
                                 zero_division=0))

    # Save final results
    results = {
        'test_accuracy': float(test_acc), 'test_f1': float(test_f1),
        'best_val_f1': float(best_f1), 'history': history,
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open('/home/user/webapp/logs/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("✅ Training complete! Results saved.")

if __name__ == '__main__':
    main()
