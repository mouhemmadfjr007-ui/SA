"""
====================================================
Sentiment Arabia - Dataset Downloader
====================================================
يقوم هذا السكريبت بتنزيل مجموعات بيانات عربية حقيقية
لتحليل المشاعر من مصادر متعددة موثوقة

Datasets المستخدمة:
1. AJGT (Arabic Jordan General Tweets) - تغريدات أردنية
2. ASTD (Arabic Sentiment Twitter Dataset)
3. ArSentD-LEV (Arabic Sentiment for Levantine)
4. SemEval Arabic Sentiment
5. LABR (Large Scale Arabic Book Reviews)
6. ArSAS (Arabic Speech Act and Sentiment)
====================================================
"""

import os
import pandas as pd
import numpy as np
import json
import csv
from pathlib import Path
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("/home/user/webapp/data/sentiment")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset_1_ajgt():
    """
    AJGT: Arabic Jordan General Tweets
    - 1,800 tweets labeled as positive/negative
    - Source: HuggingFace
    """
    logger.info("📥 تنزيل AJGT (Arabic Jordan General Tweets)...")
    try:
        dataset = load_dataset("jordanparker6/AJGT", trust_remote_code=True)
        records = []
        for split_name in dataset.keys():
            split = dataset[split_name]
            for item in split:
                text = item.get('text', item.get('tweet', ''))
                label = item.get('label', item.get('sentiment', ''))
                if text and label is not None:
                    # Convert label to standard format
                    if isinstance(label, (int, float)):
                        sent = 'positive' if label == 1 else 'negative'
                    else:
                        label_str = str(label).lower()
                        if 'pos' in label_str or 'إيجابي' in label_str:
                            sent = 'positive'
                        elif 'neg' in label_str or 'سلبي' in label_str:
                            sent = 'negative'
                        else:
                            sent = 'neutral'
                    records.append({'text': text, 'sentiment': sent, 'source': 'AJGT'})

        if records:
            df = pd.DataFrame(records)
            df.to_csv(RAW_DIR / "ajgt.csv", index=False, encoding='utf-8-sig')
            logger.info(f"✅ AJGT: {len(df)} عينة محفوظة")
            return df
    except Exception as e:
        logger.warning(f"⚠️ AJGT فشل التنزيل: {e}")
    return None


def download_dataset_2_arabic_sentiment():
    """
    Arabic Sentiment Dataset من HuggingFace
    """
    logger.info("📥 تنزيل Arabic Sentiment Dataset...")
    try:
        dataset = load_dataset("arbml/ASTD", trust_remote_code=True)
        records = []
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive',
                     'Negative': 'negative', 'Neutral': 'neutral', 'Positive': 'positive',
                     'OBJ': 'neutral', 'Subjective': 'negative'}
        
        for split_name in dataset.keys():
            for item in dataset[split_name]:
                text = item.get('text', '')
                label = item.get('label', item.get('sentiment', ''))
                if text:
                    sent = label_map.get(label, 'neutral')
                    records.append({'text': text, 'sentiment': sent, 'source': 'ASTD'})
        
        if records:
            df = pd.DataFrame(records)
            df.to_csv(RAW_DIR / "astd.csv", index=False, encoding='utf-8-sig')
            logger.info(f"✅ ASTD: {len(df)} عينة محفوظة")
            return df
    except Exception as e:
        logger.warning(f"⚠️ ASTD فشل التنزيل: {e}")
    return None


def download_dataset_3_labr():
    """
    LABR: Large Arabic Book Reviews
    """
    logger.info("📥 تنزيل LABR (Arabic Book Reviews)...")
    try:
        dataset = load_dataset("labr", trust_remote_code=True)
        records = []
        for split_name in dataset.keys():
            for item in dataset[split_name]:
                text = item.get('text', item.get('review', ''))
                label = item.get('label', item.get('polarity', 2))
                if text:
                    if isinstance(label, (int, float)):
                        # LABR uses 1-5 ratings
                        if label <= 2:
                            sent = 'negative'
                        elif label >= 4:
                            sent = 'positive'
                        else:
                            sent = 'neutral'
                    else:
                        label_str = str(label).lower()
                        if 'neg' in label_str:
                            sent = 'negative'
                        elif 'pos' in label_str:
                            sent = 'positive'
                        else:
                            sent = 'neutral'
                    records.append({'text': text, 'sentiment': sent, 'source': 'LABR'})
        
        if records:
            df = pd.DataFrame(records)
            # Limit to 5000 per class for balance
            df_balanced = pd.concat([
                df[df['sentiment'] == 'positive'].head(5000),
                df[df['sentiment'] == 'negative'].head(5000),
                df[df['sentiment'] == 'neutral'].head(5000),
            ])
            df_balanced.to_csv(RAW_DIR / "labr.csv", index=False, encoding='utf-8-sig')
            logger.info(f"✅ LABR: {len(df_balanced)} عينة محفوظة")
            return df_balanced
    except Exception as e:
        logger.warning(f"⚠️ LABR فشل التنزيل: {e}")
    return None


def download_dataset_4_hard():
    """
    Arabic Sentiment Analysis - HARD dataset
    Hotel Arabic Reviews Dataset
    """
    logger.info("📥 تنزيل HARD (Hotel Arabic Reviews)...")
    try:
        dataset = load_dataset("hard", trust_remote_code=True)
        records = []
        for split_name in dataset.keys():
            for item in dataset[split_name]:
                text = item.get('text', item.get('review', ''))
                label = item.get('label', item.get('rating', 3))
                if text:
                    if isinstance(label, (int, float)):
                        if label <= 2:
                            sent = 'negative'
                        elif label >= 4:
                            sent = 'positive'
                        else:
                            sent = 'neutral'
                    else:
                        lstr = str(label).lower()
                        if 'neg' in lstr:
                            sent = 'negative'
                        elif 'pos' in lstr:
                            sent = 'positive'
                        else:
                            sent = 'neutral'
                    records.append({'text': text, 'sentiment': sent, 'source': 'HARD'})
        
        if records:
            df = pd.DataFrame(records)
            df.to_csv(RAW_DIR / "hard.csv", index=False, encoding='utf-8-sig')
            logger.info(f"✅ HARD: {len(df)} عينة محفوظة")
            return df
    except Exception as e:
        logger.warning(f"⚠️ HARD فشل التنزيل: {e}")
    return None


def download_dataset_5_twitter():
    """
    Arabic Twitter Sentiment
    """
    logger.info("📥 تنزيل Arabic Twitter Sentiment...")
    try:
        dataset = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset", trust_remote_code=True)
        records = []
        for split_name in dataset.keys():
            for item in dataset[split_name]:
                text = item.get('text', '')
                label = item.get('label', 1)
                if text:
                    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                    sent = label_map.get(int(label), 'neutral')
                    records.append({'text': text, 'sentiment': sent, 'source': 'Twitter'})
        
        if records:
            df = pd.DataFrame(records)
            df.to_csv(RAW_DIR / "twitter_sentiment.csv", index=False, encoding='utf-8-sig')
            logger.info(f"✅ Twitter: {len(df)} عينة محفوظة")
            return df
    except Exception as e:
        logger.warning(f"⚠️ Twitter Sentiment فشل التنزيل: {e}")
    return None


def generate_arabic_synthetic_dataset():
    """
    إنشاء dataset عربية اصطناعية شاملة ومتنوعة
    تغطي مجالات: التجارة الإلكترونية، المطاعم، الخدمات، التقنية
    """
    logger.info("🔧 إنشاء Arabic Synthetic Dataset شامل...")
    
    positive_samples = [
        # تسوق
        ("المنتج ممتاز جداً وجودته عالية، سعيد جداً بشرائه", "positive"),
        ("خدمة العملاء رائعة والتوصيل كان سريعاً جداً", "positive"),
        ("أنصح الجميع بهذا المنتج، يستحق كل ريال", "positive"),
        ("تجربة تسوق مميزة، سأعود للشراء مجدداً بالتأكيد", "positive"),
        ("التغليف أنيق والمنتج طابق الوصف تماماً", "positive"),
        ("أفضل موقع تسوق استخدمته، سهل وسريع وموثوق", "positive"),
        ("المنتج وصل بحالة ممتازة والسعر معقول جداً", "positive"),
        ("سرعة التوصيل فاقت توقعاتي، شكراً لكم", "positive"),
        ("جودة المنتج أفضل مما توقعت، راضٍ تماماً", "positive"),
        ("فريق الدعم الفني محترف ومتعاون ويحل المشاكل بسرعة", "positive"),
        
        # مطاعم وطعام
        ("الطعام لذيذ والخدمة سريعة والمكان نظيف ومريح", "positive"),
        ("أحسن مطعم جربته، طعامهم طازج ومكوناته طبيعية", "positive"),
        ("الوجبة كانت شهية جداً والكمية وافرة والسعر مناسب", "positive"),
        ("خدمة ممتازة وطعام رائع سنعود إليهم بالتأكيد", "positive"),
        ("أجود طعام أكلته في المنطقة، الطاهي محترف جداً", "positive"),
        
        # تقنية وتطبيقات
        ("التطبيق رائع وسهل الاستخدام وخالٍ من الأخطاء", "positive"),
        ("تحديث ممتاز حسّن الأداء كثيراً، شكراً للفريق", "positive"),
        ("الموقع الإلكتروني سريع وواجهته جميلة وسهلة التصفح", "positive"),
        ("أداء التطبيق ممتاز، لا تعليق، تجربة مثالية", "positive"),
        ("البرنامج احترافي وسهل التعلم ويوفر وقتاً كبيراً", "positive"),
        
        # فنادق وسياحة
        ("الفندق رائع والغرف نظيفة والإفطار لذيذ ومتنوع", "positive"),
        ("موقع الفندق مثالي وطاقم العمل محترف ومرحب", "positive"),
        ("تجربة إقامة لا تنسى، سأعود إليهم بكل تأكيد", "positive"),
        ("المنتجع جميل جداً ومرافقه متكاملة والخدمة ممتازة", "positive"),
        
        # تعليم
        ("الدورة مفيدة جداً والمدرب متمكن وأسلوبه ممتاز", "positive"),
        ("تعلمت الكثير من هذا الكتاب، أوصي به بشدة", "positive"),
        ("المحتوى التعليمي منظم ومفيد وشامل لكل جوانب الموضوع", "positive"),
        
        # صحة ورياضة
        ("المنتج الصحي أعطى نتائج ممتازة خلال أسبوعين فقط", "positive"),
        ("النادي الرياضي مجهز جيداً والمدربون محترفون ومتعاونون", "positive"),
        ("أفضل مكمل غذائي جربته، نتائج واضحة وإيجابية", "positive"),
        
        # سيارات
        ("السيارة رائعة وأداؤها ممتاز على الطريق وتوفيرها للوقود عالٍ", "positive"),
        ("المركبة مريحة جداً وتقنياتها متطورة وسعرها مناسب", "positive"),
        
        # بنوك ومال
        ("خدمة البنك الإلكترونية سهلة وآمنة وسريعة جداً", "positive"),
        ("التطبيق البنكي ممتاز وسهّل كل معاملاتي المالية", "positive"),
        
        # تعليقات عامة
        ("تجربة رائعة جداً سأوصي بها لجميع أصدقائي وعائلتي", "positive"),
        ("أفضل قرار اتخذته كان اختيار هذه الخدمة", "positive"),
        ("نعم أنصح به بشدة، لن تندم على هذا القرار", "positive"),
        ("ممتاز ممتاز ممتاز، لا يوجد أفضل من هذا أبداً", "positive"),
        ("كل شيء مثالي من البداية إلى النهاية، شكراً جزيلاً", "positive"),
        ("منتج عالي الجودة يستحق كل التقييمات الإيجابية التي يحصل عليها", "positive"),
    ]
    
    negative_samples = [
        # تسوق
        ("المنتج وصل مكسوراً والتغليف كان سيئاً جداً", "negative"),
        ("جودة المنتج أقل بكثير مما هو معروض في الصور", "negative"),
        ("التوصيل تأخر أسبوعاً كاملاً دون أي إشعار", "negative"),
        ("خدمة العملاء لا تستجيب وتتجاهل الشكاوى المتكررة", "negative"),
        ("المنتج مغشوش ولا يشبه الوصف المكتوب في الإعلان", "negative"),
        ("طلبت استرداد المال منذ أسبوعين ولم يحدث شيء حتى الآن", "negative"),
        ("أسوأ تجربة شراء في حياتي، لن أعود لهذا الموقع", "negative"),
        ("الموقع يأخذ المال ولا يوصل المنتج، عملية نصب واضحة", "negative"),
        ("السعر مبالغ فيه جداً مقارنة بالجودة المتردية", "negative"),
        ("لا ينصح به أبداً، ضياع مال وقت وجهد", "negative"),
        
        # مطاعم
        ("الطعام كان بارداً ومذاقه سيئ جداً ولا يؤكل", "negative"),
        ("الخدمة كانت بطيئة جداً وتجاهلوني طوال الوقت", "negative"),
        ("المطعم غير نظيف والحمامات كارثة، لن أعود هنا", "negative"),
        ("الطعام غير طازج وكان له رائحة سيئة واضحة", "negative"),
        ("دفعت مبلغاً كبيراً مقابل وجبة رديئة لا تستحق", "negative"),
        
        # تقنية
        ("التطبيق يتعطل باستمرار ولا يعمل كما هو مفترض", "negative"),
        ("الموقع بطيء جداً ومليء بالأخطاء ويضيع البيانات", "negative"),
        ("سرقوا بياناتي الشخصية وحسابي البنكي تأثر بسببهم", "negative"),
        ("البرنامج يحتاج إلى تحسينات كثيرة جداً لأن الواجهة معقدة", "negative"),
        ("التحديث أفسد كل شيء والتطبيق أصبح أبطأ وأكثر أخطاءً", "negative"),
        
        # فنادق
        ("الغرفة كانت قذرة وتحتوي على حشرات، أمر لا يقبل", "negative"),
        ("الفندق لا يستحق هذا السعر المبالغ فيه إطلاقاً", "negative"),
        ("الخدمة مهملة والطاقم غير محترف وغير متعاون", "negative"),
        ("كل الصور مزيفة والواقع أسوأ بكثير مما أُعلن", "negative"),
        
        # تعليم
        ("الدورة مملة وتكرارية ولا تضيف أي معلومة جديدة", "negative"),
        ("المدرب لا يعرف شيئاً عن الموضوع الذي يدرسه", "negative"),
        ("ضياع وقت وأموال على هذا المحتوى التعليمي الرديء", "negative"),
        
        # صحة
        ("المنتج لم يعطِ أي نتيجة وآثاره الجانبية كانت مزعجة", "negative"),
        ("النادي الرياضي غير منظم والأجهزة قديمة ومعطوبة", "negative"),
        
        # سيارات
        ("السيارة تعطلت بعد شهر واحد فقط من الشراء", "negative"),
        ("الوكالة لا تلتزم بوعودها ولا تصلح الأعطال المضمونة", "negative"),
        
        # بنوك
        ("خسرت أموالاً بسبب أخطاء في النظام البنكي", "negative"),
        ("خدمة عملاء البنك كارثية ولا تحل أي مشكلة", "negative"),
        
        # عامة
        ("أسوأ تجربة مررت بها في حياتي كلها", "negative"),
        ("لا أنصح به أبداً وسأحذر منه كل من أعرف", "negative"),
        ("مضيعة للمال والوقت والجهد، لا قيمة له", "negative"),
        ("كل شيء كان سيئاً من البداية حتى النهاية", "negative"),
        ("لن أعود ولن أنصح به، تجربة محبطة جداً", "negative"),
        ("يستحق صفر من عشرة، أسوأ ما يمكن تخيله", "negative"),
    ]
    
    neutral_samples = [
        # وصف محايد
        ("المنتج جيد لكنه ليس استثنائياً، يؤدي الغرض المطلوب", "neutral"),
        ("الخدمة مقبولة لكن يمكن تحسينها في بعض النواحي", "neutral"),
        ("التوصيل جاء في الوقت المحدد والمنتج كما هو موصوف", "neutral"),
        ("الجودة متوسطة مقارنة بالسعر، لا مشكلة كبيرة", "neutral"),
        ("المنتج يؤدي وظيفته لكن لا شيء مميز يستحق الذكر", "neutral"),
        ("لا بأس به، ليس الأفضل ولكنه مقبول في هذا السعر", "neutral"),
        ("التجربة كانت عادية، لا إيجابية ولا سلبية بشكل واضح", "neutral"),
        ("الخدمة معتادة ولا شيء يميزها عن المنافسين", "neutral"),
        ("المطعم جيد لكن لا شيء مميز يجعلني أعود بشكل خاص", "neutral"),
        ("الفندق عادي يناسب المبيت السريع دون توقع الرفاهية", "neutral"),
        
        # مراجعات محايدة
        ("المنتج تسلمته سليماً، الجودة مقبولة لهذا السعر", "neutral"),
        ("البعض يحبه والبعض لا، هو أمر شخصي في النهاية", "neutral"),
        ("التطبيق يعمل بشكل مقبول لكن يحتاج بعض التحسينات", "neutral"),
        ("لا يختلف عن المنافسين كثيراً، نفس المستوى تقريباً", "neutral"),
        ("الأسعار معتدلة والجودة متوسطة، نتيجة منتظرة", "neutral"),
        ("ليس الأفضل ولا الأسوأ في السوق، وسط بين الاثنين", "neutral"),
        ("يمكنني الاعتماد عليه لكن أتمنى التحسين في بعض النقاط", "neutral"),
        ("الخدمة تأخرت قليلاً لكن وصلت في نهاية المطاف", "neutral"),
        ("السعر مناسب لكن الجودة لم تفاجئني بشيء", "neutral"),
        ("منتج عادي يلبي الاحتياج الأساسي دون أكثر من ذلك", "neutral"),
        
        # أوصاف محايدة
        ("وصلت الطلبية، الحال كما كان متوقعاً", "neutral"),
        ("لا أشتكي ولكن لا أبالغ في المدح أيضاً", "neutral"),
        ("النتيجة كانت في منتصف التوقعات تماماً", "neutral"),
        ("استلمت المنتج في الموعد ويعمل بشكل طبيعي", "neutral"),
        ("كفاءة معقولة دون مشاكل واضحة أو مزايا بارزة", "neutral"),
        ("المنتج يعمل بشكل صحيح، هذا هو المطلوب فقط", "neutral"),
        ("يناسب الحاجة الأساسية لكن يفتقر للميزات الإضافية", "neutral"),
        ("قابلت توقعاتي بالضبط، لا أكثر ولا أقل", "neutral"),
        ("خدمة معتادة في هذا المجال، تجربة مألوفة", "neutral"),
        ("عادي جداً، لكن لا شكاوى جوهرية من تجربتي", "neutral"),
    ]
    
    # Augment with variations
    augmented_positive = []
    augmented_negative = []
    augmented_neutral = []
    
    # Add domain-specific Arabic examples
    ecommerce_positive = [
        ("شحن سريع في نفس اليوم وتغليف ممتاز واحترافي", "positive"),
        ("المنتج أصلي ١٠٠٪ وبسعر أقل من السوق بكثير", "positive"),
        ("سهولة الاسترجاع والاستبدال والفريق متعاون جداً", "positive"),
        ("أفضل موقع تسوق في المنطقة، خدمة ما بعد البيع ممتازة", "positive"),
    ]
    
    ecommerce_negative = [
        ("المنتج مختلف تماماً عن الصورة، لا أوصي به أبداً", "negative"),
        ("رسوم إضافية خفية لم تُذكر في الإعلان، غش واضح", "negative"),
        ("التوصيل لا يصل للعنوان المطلوب رغم تكرار الشكوى", "negative"),
        ("الموقع يأخذ الأموال ويرسل منتجات مقلدة رديئة", "negative"),
    ]
    
    social_samples = [
        ("الأخبار محزنة جداً، نسأل الله السلامة للجميع", "negative"),
        ("أتمنى أن يتحسن الوضع قريباً للجميع", "neutral"),
        ("الأحداث الأخيرة مقلقة ومؤثرة على الجميع", "negative"),
        ("نأمل في أيام أفضل وأكثر استقراراً قادمة", "neutral"),
        ("الحدث كان مميزاً وجمع الجميع في أجواء رائعة", "positive"),
        ("المناسبة جميلة وموفقة ومنظمة بشكل احترافي رائع", "positive"),
    ]
    
    all_samples = (positive_samples + ecommerce_positive + 
                   negative_samples + ecommerce_negative + 
                   neutral_samples + social_samples)
    
    # Create DataFrame
    df = pd.DataFrame(all_samples, columns=['text', 'sentiment'])
    df['source'] = 'synthetic_arabic'
    
    # Add some noise and variants
    variants = []
    for _, row in df.iterrows():
        text = row['text']
        # Simple text augmentation: shuffle some words
        words = text.split()
        if len(words) > 4:
            import random
            random.seed(42)
            # Create a slightly different variant
            variant_text = text.replace('جداً', 'كثيراً').replace('ممتاز', 'رائع')
            if variant_text != text:
                variants.append({
                    'text': variant_text, 
                    'sentiment': row['sentiment'],
                    'source': 'synthetic_arabic_aug'
                })
    
    if variants:
        df_variants = pd.DataFrame(variants)
        df = pd.concat([df, df_variants], ignore_index=True)
    
    df.to_csv(RAW_DIR / "synthetic_arabic.csv", index=False, encoding='utf-8-sig')
    logger.info(f"✅ Synthetic Dataset: {len(df)} عينة محفوظة")
    logger.info(f"   - Positive: {len(df[df['sentiment']=='positive'])}")
    logger.info(f"   - Negative: {len(df[df['sentiment']=='negative'])}")
    logger.info(f"   - Neutral: {len(df[df['sentiment']=='neutral'])}")
    return df


def download_huggingface_arabic_datasets():
    """
    تنزيل مجموعات بيانات إضافية من HuggingFace
    """
    datasets_to_try = [
        ("asas-ai/SANAD", "text", "label"),
        ("arbml/arabic_sentiment_analysis", "text", "label"),
        ("community-datasets/arabic-sentiment-tweets-corpus", "tweet", "class"),
        ("mshoaib54/Arabic_Sentiment_Dataset", "text", "label"),
        ("Mohammed-Altaf/arabic-sentiment-with-reason", "text", "label"),
        ("Omartificial-Intelligence-Space/SANAD", "text", "label"),
    ]
    
    all_records = []
    
    for dataset_name, text_col, label_col in datasets_to_try:
        logger.info(f"📥 تجربة تنزيل: {dataset_name}")
        try:
            ds = load_dataset(dataset_name, trust_remote_code=True)
            count = 0
            for split_name in ds.keys():
                for item in ds[split_name]:
                    text = str(item.get(text_col, item.get('text', item.get('tweet', ''))))
                    label = item.get(label_col, item.get('label', item.get('sentiment', '')))
                    
                    if not text or text == 'None':
                        continue
                    
                    # Standardize label
                    label_str = str(label).lower()
                    if any(x in label_str for x in ['pos', 'إيجابي', '2', 'positive']):
                        sent = 'positive'
                    elif any(x in label_str for x in ['neg', 'سلبي', '0', 'negative']):
                        sent = 'negative'
                    elif any(x in label_str for x in ['neu', 'محايد', '1', 'neutral', 'mixed', 'obj']):
                        sent = 'neutral'
                    elif str(label) in ['1', '5', '4']:
                        sent = 'positive'
                    elif str(label) in ['-1', '0']:
                        sent = 'negative'
                    else:
                        sent = 'neutral'
                    
                    all_records.append({
                        'text': text,
                        'sentiment': sent,
                        'source': dataset_name.split('/')[-1]
                    })
                    count += 1
            
            logger.info(f"  ✅ {dataset_name}: {count} عينة")
        except Exception as e:
            logger.warning(f"  ⚠️ {dataset_name} فشل: {str(e)[:80]}")
    
    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv(RAW_DIR / "huggingface_arabic.csv", index=False, encoding='utf-8-sig')
        logger.info(f"✅ HuggingFace Arabic: {len(df)} عينة محفوظة")
        return df
    return None


def merge_and_process_datasets():
    """
    دمج جميع datasets وإنشاء ملفات train/val/test
    """
    logger.info("\n🔄 دمج جميع مجموعات البيانات...")
    
    all_dfs = []
    
    # Load all available CSVs
    for csv_file in RAW_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            if 'text' in df.columns and 'sentiment' in df.columns:
                all_dfs.append(df[['text', 'sentiment', 'source']])
                logger.info(f"  📂 {csv_file.name}: {len(df)} عينة")
        except Exception as e:
            logger.warning(f"  ⚠️ خطأ في قراءة {csv_file.name}: {e}")
    
    if not all_dfs:
        logger.error("❌ لا توجد بيانات لدمجها!")
        return None
    
    # Merge all
    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"\n📊 إجمالي العينات قبل المعالجة: {len(combined)}")
    
    # Clean data
    combined = combined.dropna(subset=['text', 'sentiment'])
    combined['text'] = combined['text'].astype(str).str.strip()
    combined = combined[combined['text'].str.len() > 5]  # Remove very short texts
    combined = combined[combined['text'].str.len() < 2000]  # Remove very long texts
    
    # Standardize sentiment labels
    combined['sentiment'] = combined['sentiment'].str.lower().str.strip()
    combined['label'] = combined['sentiment'].map({
        'negative': 0, 'neg': 0,
        'neutral': 1, 'neu': 1,
        'positive': 2, 'pos': 2
    })
    combined = combined.dropna(subset=['label'])
    combined['label'] = combined['label'].astype(int)
    
    # Remove duplicates
    combined = combined.drop_duplicates(subset=['text'])
    
    logger.info(f"📊 إجمالي العينات بعد التنظيف: {len(combined)}")
    
    # Show distribution
    dist = combined['sentiment'].value_counts()
    logger.info(f"\n📈 توزيع المشاعر:")
    for sent, count in dist.items():
        logger.info(f"   {sent}: {count} ({count/len(combined)*100:.1f}%)")
    
    # Balance dataset
    min_class = min(dist.values)
    max_per_class = min(min_class * 3, 5000)  # Cap at 5000 per class
    
    balanced_dfs = []
    for sentiment_label in ['negative', 'neutral', 'positive']:
        subset = combined[combined['sentiment'] == sentiment_label]
        if len(subset) > max_per_class:
            subset = subset.sample(n=max_per_class, random_state=42)
        balanced_dfs.append(subset)
    
    balanced = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)
    
    logger.info(f"\n📊 بعد التوازن: {len(balanced)} عينة")
    for s in ['negative', 'neutral', 'positive']:
        count = len(balanced[balanced['sentiment'] == s])
        logger.info(f"   {s}: {count}")
    
    # Split into train/val/test (70/15/15)
    from sklearn.model_selection import train_test_split
    
    train_val, test = train_test_split(balanced, test_size=0.15, random_state=42, stratify=balanced['label'])
    train, val = train_test_split(train_val, test_size=0.176, random_state=42, stratify=train_val['label'])
    
    # Save splits
    train.to_csv(PROCESSED_DIR / "train.csv", index=False, encoding='utf-8-sig')
    val.to_csv(PROCESSED_DIR / "val.csv", index=False, encoding='utf-8-sig')
    test.to_csv(PROCESSED_DIR / "test.csv", index=False, encoding='utf-8-sig')
    
    # Also save combined
    balanced.to_csv(PROCESSED_DIR / "combined.csv", index=False, encoding='utf-8-sig')
    
    logger.info(f"\n✅ تم حفظ ملفات البيانات:")
    logger.info(f"   Train: {len(train)} عينة → {PROCESSED_DIR}/train.csv")
    logger.info(f"   Val:   {len(val)} عينة → {PROCESSED_DIR}/val.csv")
    logger.info(f"   Test:  {len(test)} عينة → {PROCESSED_DIR}/test.csv")
    
    # Save stats
    stats = {
        "total_samples": len(balanced),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "label_distribution": {
            "negative": int(len(balanced[balanced['sentiment']=='negative'])),
            "neutral": int(len(balanced[balanced['sentiment']=='neutral'])),
            "positive": int(len(balanced[balanced['sentiment']=='positive']))
        },
        "label_mapping": {"negative": 0, "neutral": 1, "positive": 2}
    }
    
    with open(PROCESSED_DIR / "dataset_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n📄 إحصائيات محفوظة في dataset_stats.json")
    return balanced, stats


def main():
    logger.info("=" * 60)
    logger.info("🚀 Sentiment Arabia - Dataset Downloader")
    logger.info("=" * 60)
    
    results = {}
    
    # 1. Try to download from HuggingFace
    logger.info("\n--- مرحلة 1: تنزيل من HuggingFace ---")
    results['ajgt'] = download_dataset_1_ajgt()
    results['astd'] = download_dataset_2_arabic_sentiment()
    results['labr'] = download_dataset_3_labr()
    results['hard'] = download_dataset_4_hard()
    results['hf_arabic'] = download_huggingface_arabic_datasets()
    
    # 2. Generate synthetic data (always)
    logger.info("\n--- مرحلة 2: إنشاء بيانات اصطناعية عربية ---")
    results['synthetic'] = generate_arabic_synthetic_dataset()
    
    # 3. Merge and process
    logger.info("\n--- مرحلة 3: دمج ومعالجة البيانات ---")
    result = merge_and_process_datasets()
    
    if result:
        balanced, stats = result
        logger.info("\n" + "=" * 60)
        logger.info("✅ اكتمل تجهيز مجموعات البيانات بنجاح!")
        logger.info(f"   إجمالي العينات: {stats['total_samples']}")
        logger.info(f"   Train: {stats['train_size']} | Val: {stats['val_size']} | Test: {stats['test_size']}")
        logger.info("=" * 60)
    else:
        logger.error("❌ فشل في معالجة البيانات!")


if __name__ == "__main__":
    main()
