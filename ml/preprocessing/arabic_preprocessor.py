"""
====================================================
Sentiment Arabia - Data Preprocessing Pipeline
====================================================
معالجة وتنظيف البيانات العربية لتحليل المشاعر
====================================================
"""

import re
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ArabicTextPreprocessor:
    """
    معالج النصوص العربية
    يقوم بتنظيف وتوحيد النصوص العربية للاستخدام في نماذج NLP
    """
    
    # Arabic Unicode ranges
    ARABIC_CHARS = r'\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF'
    
    # Common Arabic stopwords (but keep them for sentiment analysis context)
    SENTIMENT_STOPWORDS = set()  # Don't remove stopwords for sentiment
    
    def __init__(self, 
                 remove_diacritics: bool = True,
                 normalize_arabic: bool = True,
                 remove_urls: bool = True,
                 remove_mentions: bool = True,
                 remove_hashtags: bool = False,  # Keep hashtags - may contain sentiment
                 remove_emojis: bool = False,    # Keep emojis - strong sentiment signals
                 max_length: int = 512):
        
        self.remove_diacritics = remove_diacritics
        self.normalize_arabic = normalize_arabic
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_emojis = remove_emojis
        self.max_length = max_length
    
    def normalize_unicode(self, text: str) -> str:
        """تطبيع Unicode"""
        return unicodedata.normalize('NFC', text)
    
    def remove_diacritics_func(self, text: str) -> str:
        """إزالة التشكيل العربي"""
        # Arabic diacritics (tashkeel)
        diacritics = re.compile(r'[\u064B-\u065F\u0670]')
        return diacritics.sub('', text)
    
    def normalize_arabic_chars(self, text: str) -> str:
        """توحيد الحروف العربية المتشابهة"""
        # Normalize alef variants
        text = re.sub(r'[أإآا]', 'ا', text)
        # Normalize hamza
        text = re.sub(r'[ؤئء]', 'ء', text)
        # Normalize teh marbuta
        text = re.sub(r'ة', 'ه', text)
        # Normalize ya
        text = re.sub(r'[يى]', 'ي', text)
        # Normalize kaf
        text = re.sub(r'ك', 'ك', text)
        return text
    
    def clean_text(self, text: str) -> str:
        """تنظيف النص الأساسي"""
        if not isinstance(text, str):
            return ""
        
        # Normalize unicode
        text = self.normalize_unicode(text)
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            text = re.sub(r'www\.[^\s]+', '', text)
        
        # Remove mentions
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags or keep text only
        if self.remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        else:
            text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag text
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove diacritics
        if self.remove_diacritics:
            text = self.remove_diacritics_func(text)
        
        # Normalize Arabic characters
        if self.normalize_arabic:
            text = self.normalize_arabic_chars(text)
        
        # Remove non-Arabic and non-essential chars (keep punctuation and digits)
        # Keep: Arabic chars, digits, spaces, basic punctuation
        text = re.sub(
            r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF'
            r'\u2600-\u26FF\u2700-\u27BF'  # emoji blocks
            r'\U0001F300-\U0001F9FF'        # more emoji
            r'0-9\s\.,!؟?،؛:«»\-]',
            ' ', text
        )
        
        # Clean up spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate if too long
        if self.max_length and len(text) > self.max_length * 4:  # chars not tokens
            text = text[:self.max_length * 4]
        
        return text
    
    def process(self, text: str) -> str:
        """المعالجة الكاملة للنص"""
        cleaned = self.clean_text(text)
        return cleaned
    
    def process_batch(self, texts: list) -> list:
        """معالجة مجموعة من النصوص"""
        return [self.process(text) for text in texts]


def preprocess_dataset(input_path: str, output_path: str = None, 
                       min_length: int = 3, max_length: int = 512) -> pd.DataFrame:
    """
    معالجة ملف CSV كامل من البيانات
    
    Args:
        input_path: مسار ملف CSV الإدخال
        output_path: مسار ملف CSV الإخراج (اختياري)
        min_length: الحد الأدنى لطول النص (بالكلمات)
        max_length: الحد الأقصى لطول النص (بالأحرف)
    
    Returns:
        DataFrame مع النصوص المعالجة
    """
    logger.info(f"📂 قراءة: {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    
    preprocessor = ArabicTextPreprocessor(max_length=max_length)
    
    # Process texts
    logger.info("🔄 معالجة النصوص...")
    df['text_clean'] = df['text'].apply(preprocessor.process)
    
    # Filter by length
    original_count = len(df)
    df = df[df['text_clean'].str.split().str.len() >= min_length]
    df = df[df['text_clean'].str.len() > 0]
    df = df.dropna(subset=['text_clean'])
    
    removed = original_count - len(df)
    if removed > 0:
        logger.info(f"🗑️ تمت إزالة {removed} عينة قصيرة/فارغة")
    
    logger.info(f"✅ النصوص المعالجة: {len(df)}")
    
    if output_path:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"💾 محفوظ في: {output_path}")
    
    return df


def get_data_statistics(df: pd.DataFrame) -> dict:
    """
    إحصائيات عن مجموعة البيانات
    """
    stats = {
        "total": len(df),
        "by_sentiment": df['sentiment'].value_counts().to_dict(),
        "text_lengths": {
            "mean": float(df['text'].str.len().mean()),
            "median": float(df['text'].str.len().median()),
            "min": int(df['text'].str.len().min()),
            "max": int(df['text'].str.len().max()),
        },
        "word_counts": {
            "mean": float(df['text'].str.split().str.len().mean()),
            "median": float(df['text'].str.split().str.len().median()),
        }
    }
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    DATA_DIR = Path("/home/user/webapp/data/sentiment/processed")
    
    for split in ['train', 'val', 'test']:
        input_file = DATA_DIR / f"{split}.csv"
        if input_file.exists():
            df = preprocess_dataset(
                str(input_file),
                str(DATA_DIR / f"{split}_clean.csv")
            )
            stats = get_data_statistics(df)
            logger.info(f"\n📊 إحصائيات {split}: {stats}")
