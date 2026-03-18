# Sentiment Arabia 🌙

نظام تحليل المشاعر العربي للشركات - Arabic Sentiment Analysis System

## نظرة عامة

**Sentiment Arabia** هو نظام متكامل لتحليل مشاعر النصوص العربية مبني على نموذج ArabicBERT، يستهدف الشركات ويدعم تحليل المنتجات والخدمات.

## الميزات الرئيسية

| الميزة | الوصف |
|--------|-------|
| 🔍 تصنيف المشاعر | سلبي / محايد / إيجابي بدقة عالية |
| ⚡ أداء عالي | زمن استجابة < 200ms مع LRU Cache |
| 📦 تحليل دفعي | حتى 100 نص في طلب واحد |
| 🏷️ استخراج الجوانب | جودة، توصيل، سعر، خدمة عملاء... |
| 🚨 كشف الاحتيال | كشف النصوص الاحتيالية والمزيفة |
| 🏢 دعم الشركات | إدارة شركات ومنتجات متعددة |
| 🤖 نموذج هجين | BERT + قواعد عربية متقدمة |

## البنية التقنية

```
Sentiment Arabia/
├── api/                    # FastAPI Backend
│   ├── main.py            # نقاط API الرئيسية
│   └── v1/routes/         # مسارات API
├── ml/inference/           # محرك الاستدلال
│   └── optimized_inference_engine.py
├── training/               # سكريبتات التدريب
│   ├── train_final.py     # التدريب الكامل
│   └── train_sentiment_pytorch.py
├── models/sentiment/       # النماذج المحفوظة
│   └── best_model/
├── data/sentiment/         # البيانات
│   └── processed/         # train/val/test.csv
├── frontend/public/        # الواجهة الأمامية
│   └── index.html
├── tests/                  # الاختبارات
│   └── test_sentiment_comprehensive.py
└── manage.py              # أداة الإدارة
```

## التثبيت والتشغيل

### 1. تثبيت المتطلبات
```bash
pip install fastapi uvicorn transformers torch datasets pandas scikit-learn redis httpx pydantic
```

### 2. تشغيل الخادم
```bash
python manage.py runserver --port 8000
# أو مباشرة:
cd api && python main.py
```

### 3. التدريب
```bash
python manage.py train --max_samples 2000 --epochs 4 --batch_size 8
# أو مباشرة:
python training/train_final.py --max_samples 2000 --epochs 4
```

### 4. فحص الحالة
```bash
python manage.py status
```

## نقاط API

| المسار | الطريقة | الوصف |
|--------|---------|-------|
| `/api/v1/analyze` | POST | تحليل نص واحد |
| `/api/v1/analyze/batch` | POST | تحليل دفعي |
| `/api/v1/analyze/demo` | POST | عرض توضيحي |
| `/api/v1/health` | GET | فحص الصحة |
| `/api/v1/stats` | GET | الإحصائيات |
| `/api/v1/companies` | POST/GET | إدارة الشركات |
| `/api/v1/history` | GET | سجل التحليلات |

## مثال استخدام API

```python
import httpx

# تحليل نص واحد
response = httpx.post("http://localhost:8000/api/v1/analyze", json={
    "text": "المنتج ممتاز جداً وأنصح به",
    "include_aspects": True,
    "include_fraud": True
})
print(response.json())
# {"sentiment": "positive", "sentiment_ar": "إيجابي", "confidence": 0.89, ...}
```

## النموذج

- **Base**: `aubmindlab/bert-base-arabertv02` (ArabicBERT)
- **Fine-tuning**: على 10,500+ نص عربي (LABR + HARD)
- **Labels**: negative(0), neutral(1), positive(2)
- **Approach**: هجين BERT + قواعد عربية

## البيانات

| المجموعة | المصدر | الحجم |
|----------|--------|-------|
| Train | LABR + HARD | 10,506 |
| Val | LABR + HARD | 2,244 |
| Test | LABR + HARD | 2,250 |

## أهداف الأداء

- ✅ دقة ≥ 90%
- ✅ زمن استجابة < 200ms
- ✅ P95 < 100ms (مع كاش)
- ✅ 50+ طلب/ثانية
- ✅ 99%+ uptime

## المكتبات المستخدمة

```
fastapi, uvicorn, transformers, torch, datasets
pandas, scikit-learn, redis, httpx, pydantic
```

---

**Sentiment Arabia** - تحليل المشاعر العربية بدقة واحترافية 🌙
