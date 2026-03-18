"""
Sentiment Arabia - Optimized Inference Engine v2
محرك استدلال محسّن مع LRU cache + هجين BERT + قواعد عربية
"""
import os, re, json, time, hashlib, logging
from pathlib import Path
from typing import List, Dict
from collections import OrderedDict
from threading import Lock

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

LABEL_MAP    = {0: 'negative', 1: 'neutral', 2: 'positive'}
LABEL_MAP_AR = {0: 'سلبي', 1: 'محايد', 2: 'إيجابي'}
MODEL_DIR    = Path("/home/user/webapp/models/sentiment/best_model")
FALLBACK_MDL = "aubmindlab/bert-base-arabertv02"

POSITIVE_WORDS = {
    'ممتاز','رائع','جيد','مميز','جميل','أحسن','أفضل','شكراً','شكرا',
    'أحب','أنصح','يستحق','مبهج','سعيد','فرحان','مرضٍ','مرضي','ناجح',
    'نجح','ممتازة','رائعة','مميزة','جميلة','سريع','دقيق','موثوق',
    'احترافي','محترف','لذيذ','شهي','نظيف','منظم','ملتزم','أوصي',
    'مذهل','استثنائي','عالي','راضٍ','راضي','مسرور','مبهور','خارق',
    'الأفضل','أجمل','أروع','أمتع','أسرع','ممتع','رائق','بديع','فاخر'
}
NEGATIVE_WORDS = {
    'سيء','رديء','فاشل','كارثة','أسوأ','مشكلة','لا أنصح','غلط',
    'خطأ','مزيف','مقلد','متأخر','تأخر','بطيء','غبن','سرقة','نصب',
    'خداع','تضليل','مبالغ','مكسور','تالف','معطوب','لا يعمل',
    'مخيب','خيبة','إحباط','محبط','مؤلم','تعب','خسارة','ضياع',
    'فشل','مضيعة','رفض','يرفض','إرجاع','استرداد','مزعج','مضايق',
    'غضب','غاضب','قلق','خطر','ضار','وحش','قبيح','بشع','تعيس',
    'حزين','لا أوصي','لن أعود','آخر مرة','لا أشتري','كذب','ضعيف'
}
NEUTRAL_WORDS = {
    'عادي','مقبول','كافٍ','كافي','يكفي','لا بأس','مناسب','وسط',
    'متوسط','أحياناً','أحيانا','نوعاً ما','نوعا ما','يمكن','ربما',
    'يستحق المحاولة','بين بين','لا شيء مميز','ليس الأفضل','معقول'
}
NEGATION_WORDS = {'لا','لن','ليس','لم','غير','مش','مو','ما','بدون'}
INTENSIFIERS   = {'جداً','جدا','كثيراً','كثيرا','للغاية','تماماً','تماما','جدًا'}

FRAUD_SIGNALS = {
    'high':   ['مزيف','مقلد','نصب','احتيال','غش','خداع','تضليل','وهمي','سرقة','نصاب','محتال','غير أصلي'],
    'medium': ['لم يصل','اختلف عن الوصف','صورة مختلفة','الجودة رديئة','لا يعمل','معطوب','تالف','مكسور','مستعمل'],
    'low':    ['متأخر','تأخير','بطيء','لم أستلم','سؤ خدمة','مبالغ بالسعر']
}

ASPECTS = {
    'جودة_المنتج':  ['جودة','مواد','خامة','متانة','تصنيع','صنع','مصنوع'],
    'التوصيل':      ['توصيل','شحن','وصول','استلام','تأخر','سريع','بطيء'],
    'السعر':        ['سعر','تكلفة','غالي','رخيص','مبالغ','قيمة','يستحق'],
    'خدمة_العملاء': ['خدمة','دعم','تواصل','رد','مساعدة','موظف','احترافية'],
    'التغليف':      ['تغليف','كرتون','صندوق','حماية','غلاف','علبة'],
    'المظهر':       ['شكل','مظهر','لون','جميل','قبيح','تصميم','أنيق'],
    'الأداء':       ['أداء','سرعة','كفاءة','قوة','يعمل','لا يعمل','فعّال']
}


def rule_based_sentiment(text: str) -> dict:
    words = text.split()
    pos_score = neg_score = neu_score = 0.0
    for i, word in enumerate(words):
        w = word.strip('.,!؟?،؛:«»""')
        prev = words[i-1].strip('.,!؟?') if i > 0 else ''
        is_negated = prev in NEGATION_WORDS
        intensifier = 1.5 if (prev in INTENSIFIERS or (i > 1 and words[i-2] in INTENSIFIERS)) else 1.0
        if w in POSITIVE_WORDS:
            if is_negated: neg_score += 1.2 * intensifier
            else: pos_score += 1.0 * intensifier
        elif w in NEGATIVE_WORDS:
            if is_negated: pos_score += 0.8 * intensifier
            else: neg_score += 1.0 * intensifier
        elif w in NEUTRAL_WORDS:
            neu_score += 0.8
    pos_emoji = len(re.findall(r'[😊😄👍✅🌟⭐💯🎉😍🥰💪👏🔥✨]', text))
    neg_emoji = len(re.findall(r'[😞😠👎❌💔😡🤬😤😢😭💢]', text))
    pos_score += pos_emoji * 1.5
    neg_score += neg_emoji * 1.5
    total = pos_score + neg_score + neu_score
    if total == 0:
        return {'label': 1, 'confidence': 0.40, 'method': 'rule_neutral',
                'probabilities': {'negative': 0.3, 'neutral': 0.4, 'positive': 0.3}}
    if pos_score >= neg_score and pos_score >= neu_score:
        label = 2; conf = min(0.55 + (pos_score - neg_score) / (total + 1) * 0.3, 0.88)
        p = [0.1, 0.15, conf]
    elif neg_score > pos_score and neg_score >= neu_score:
        label = 0; conf = min(0.55 + (neg_score - pos_score) / (total + 1) * 0.3, 0.88)
        p = [conf, 0.15, 0.1]
    else:
        label = 1; conf = 0.50; p = [0.2, conf, 0.2]
    s = sum(p)
    return {'label': label, 'confidence': float(conf), 'method': 'rule',
            'probabilities': {'negative': p[0]/s, 'neutral': p[1]/s, 'positive': p[2]/s}}


def extract_aspects(text: str, sentiment: str) -> List[dict]:
    found = []
    for aspect, keywords in ASPECTS.items():
        if any(kw in text for kw in keywords):
            found.append({'aspect': aspect, 'sentiment': sentiment})
    return found


def detect_fraud(text: str) -> dict:
    signals = []
    risk_level = 'none'
    for level, words in FRAUD_SIGNALS.items():
        for w in words:
            if w in text:
                signals.append({'signal': w, 'level': level})
                if level == 'high': risk_level = 'high'
                elif level == 'medium' and risk_level not in ('high',): risk_level = 'medium'
                elif level == 'low' and risk_level == 'none': risk_level = 'low'
    return {
        'is_fraud_risk': len(signals) > 0,
        'risk_level': risk_level,
        'signals': signals[:5],
        'fraud_score': round(min(len(signals) * 0.2, 1.0), 2)
    }


class LRUCache:
    def __init__(self, capacity=3000, ttl=7200):
        self.cap = capacity; self.ttl = ttl
        self.cache = OrderedDict(); self.ts = {}
        self.lock = Lock(); self.hits = self.misses = 0

    def get(self, key):
        with self.lock:
            if key not in self.cache or time.time() - self.ts[key] > self.ttl:
                self.misses += 1; return None
            self.cache.move_to_end(key); self.hits += 1
            return dict(self.cache[key])

    def set(self, key, val):
        with self.lock:
            if len(self.cache) >= self.cap:
                k = next(iter(self.cache)); del self.cache[k], self.ts[k]
            self.cache[key] = val; self.ts[key] = time.time()

    @property
    def hit_rate(self):
        t = self.hits + self.misses; return self.hits / t if t else 0.0

    def stats(self):
        return {'size': len(self.cache), 'capacity': self.cap,
                'hits': self.hits, 'misses': self.misses, 'hit_rate': round(self.hit_rate, 4)}


class SentimentInferenceEngine:
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.model = self.tokenizer = self.device = None
        self.model_loaded = False
        self.cache = LRUCache()
        self.max_length = 128
        self.total_requests = 0
        self.total_latency  = 0.0
        self.load_time = None
        self._load_model()

    def _load_model(self):
        t0 = time.time()
        path = str(MODEL_DIR) if (MODEL_DIR / "config.json").exists() else FALLBACK_MDL
        logger.info(f"Loading model from: {path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path, num_labels=3, ignore_mismatched_sizes=True)
            self.device = torch.device('cpu')
            self.model.to(self.device).eval()
            self.model_loaded = True
            self.load_time = time.time() - t0
            logger.info(f"Model loaded in {self.load_time:.2f}s")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self.model_loaded = False

    def _clean(self, text: str) -> str:
        if not isinstance(text, str): return ""
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _bert_predict(self, text: str) -> dict:
        enc = self.tokenizer(text, truncation=True, padding=True,
                             max_length=self.max_length, return_tensors='pt')
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, -1)[0].numpy()
        label = int(np.argmax(probs))
        return {
            'label': label, 'confidence': float(probs[label]), 'method': 'bert',
            'probabilities': {'negative': float(probs[0]), 'neutral': float(probs[1]), 'positive': float(probs[2])}
        }

    def _hybrid_predict(self, text: str) -> dict:
        rule = rule_based_sentiment(text)
        if not self.model_loaded:
            return rule
        bert = self._bert_predict(text)
        if bert['confidence'] >= 0.65:
            return bert
        elif bert['confidence'] >= 0.45:
            alpha = 0.60
            probs = {k: alpha * bert['probabilities'].get(k, 0.33) +
                        (1 - alpha) * rule['probabilities'].get(k, 0.33)
                     for k in ('negative', 'neutral', 'positive')}
            label_str = max(probs, key=probs.get)
            lmap = {'negative': 0, 'neutral': 1, 'positive': 2}
            return {'label': lmap[label_str], 'confidence': float(probs[label_str]),
                    'method': 'hybrid', 'probabilities': probs}
        else:
            return rule if rule['confidence'] > bert['confidence'] else bert

    def predict(self, text: str, include_aspects: bool = False, include_fraud: bool = False) -> dict:
        t0 = time.time()
        text = self._clean(text)
        if not text:
            return {'error': 'النص فارغ', 'sentiment': 'neutral', 'sentiment_ar': 'محايد',
                    'confidence': 0.0, 'latency_ms': 0}
        cache_key = self._key(text)
        cached = self.cache.get(cache_key)
        if cached:
            cached['from_cache'] = True
            cached['latency_ms'] = round((time.time() - t0) * 1000, 2)
            self.total_requests += 1
            return cached
        pred = self._hybrid_predict(text)
        label = pred['label']
        result = {
            'text_preview': text[:80] + ('...' if len(text) > 80 else ''),
            'sentiment':    LABEL_MAP[label],
            'sentiment_ar': LABEL_MAP_AR[label],
            'label':        label,
            'confidence':   round(pred['confidence'], 4),
            'method':       pred.get('method', 'unknown'),
            'probabilities': {k: round(v, 4) for k, v in pred['probabilities'].items()},
            'from_cache':   False
        }
        if include_aspects:
            result['aspects'] = extract_aspects(text, LABEL_MAP[label])
        if include_fraud:
            result['fraud_detection'] = detect_fraud(text)
        latency = (time.time() - t0) * 1000
        result['latency_ms'] = round(latency, 2)
        self.cache.set(cache_key, result)
        self.total_requests += 1
        self.total_latency  += latency
        return result

    def predict_batch(self, texts: List[str], include_aspects: bool = False,
                      include_fraud: bool = False) -> List[dict]:
        return [self.predict(t, include_aspects=include_aspects,
                             include_fraud=include_fraud) for t in texts]

    @property
    def avg_latency(self):
        return self.total_latency / self.total_requests if self.total_requests else 0

    def stats(self) -> dict:
        return {
            'total_requests': self.total_requests,
            'avg_latency_ms': round(self.avg_latency, 2),
            'model_loaded':   self.model_loaded,
            'load_time_s':    round(self.load_time or 0, 2),
            'cache':          self.cache.stats()
        }


def get_engine() -> SentimentInferenceEngine:
    return SentimentInferenceEngine.get_instance()
