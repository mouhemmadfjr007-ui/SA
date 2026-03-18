"""
Sentiment Arabia - Project Manager
سكريبت إدارة المشروع: تشغيل، اختبار، تدريب
"""
import os, sys, argparse, subprocess
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def run_server(port=8000, reload=False):
    """تشغيل خادم FastAPI"""
    print(f"🚀 Starting Sentiment Arabia API on port {port}...")
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--log-level", "info",
        "--workers", "1"
    ]
    if reload:
        cmd.append("--reload")
    os.chdir(ROOT)
    subprocess.run(cmd)


def run_training(max_samples=2000, epochs=4, batch_size=8, lr=2e-5, patience=3):
    """تشغيل التدريب"""
    print(f"🔥 Starting training: {max_samples} samples, {epochs} epochs")
    cmd = [
        sys.executable, "training/train_final.py",
        "--max_samples", str(max_samples),
        "--epochs",      str(epochs),
        "--batch_size",  str(batch_size),
        "--lr",          str(lr),
        "--patience",    str(patience)
    ]
    os.chdir(ROOT)
    subprocess.run(cmd)


def run_tests():
    """تشغيل الاختبارات الشاملة"""
    print("🧪 Running comprehensive tests...")
    os.chdir(ROOT)
    subprocess.run([sys.executable, "tests/test_sentiment_comprehensive.py"])


def check_status():
    """فحص حالة المشروع"""
    print("\n" + "="*50)
    print("📊 Sentiment Arabia - Project Status")
    print("="*50)

    # Model
    model_dir = ROOT / "models/sentiment/best_model"
    config = model_dir / "config.json"
    ck = model_dir / "checkpoint_info.json"
    if config.exists():
        print(f"✅ Model: {model_dir}")
        if ck.exists():
            import json
            info = json.load(open(ck))
            print(f"   Val Acc: {info.get('val_acc',0):.3f}  Val F1: {info.get('val_f1',0):.3f}")
    else:
        print("❌ Model: not found")

    # Dataset
    data_dir = ROOT / "data/sentiment/processed"
    for f in ["train.csv", "val.csv", "test.csv"]:
        p = data_dir / f
        if p.exists():
            import pandas as pd
            df = pd.read_csv(p)
            print(f"✅ {f}: {len(df)} samples")
        else:
            print(f"❌ {f}: not found")

    # Training results
    tr = ROOT / "logs/training_results.json"
    if tr.exists():
        import json
        res = json.load(open(tr))
        print(f"\n📈 Last Training Results:")
        print(f"   Test Accuracy: {res.get('test_accuracy',0):.4f}")
        print(f"   Test F1:       {res.get('test_f1',0):.4f}")

    print("="*50 + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sentiment Arabia Manager")
    p.add_argument("command", choices=["runserver","train","test","status"],
                   help="الأمر المطلوب")
    p.add_argument("--port",        type=int, default=8000)
    p.add_argument("--reload",      action="store_true")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--epochs",      type=int, default=4)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--patience",    type=int, default=3)

    args = p.parse_args()

    if args.command == "runserver":
        run_server(args.port, args.reload)
    elif args.command == "train":
        run_training(args.max_samples, args.epochs, args.batch_size, args.lr, args.patience)
    elif args.command == "test":
        run_tests()
    elif args.command == "status":
        check_status()
