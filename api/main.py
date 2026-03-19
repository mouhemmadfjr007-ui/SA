"""
Sentiment Arabia - FastAPI v4 (Complete Multi-Tenant Platform)
منصة متكاملة: تحليل مشاعر + بحث + مقارنة + Webhooks + Dashboard + كشف احتيال
"""
import os, sys, json, time, hmac, hashlib, logging, secrets, re
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, Header, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import (
    init_db, seed_demo_data, get_db,
    get_company, get_all_companies, authenticate_user, create_company,
    get_company_by_api_key,
    get_product, search_products, get_company_products, create_product,
    save_analysis, get_product_analyses, get_company_analyses,
    get_product_sentiment_stats, get_company_dashboard_stats, get_market_insights,
    create_webhook, get_webhook_config, record_webhook_call,
    get_products_for_compare, get_fraud_alerts
)
from api.auth import create_token, get_current_company, get_current_company_optional, check_company_permission
from ml.inference.optimized_inference_engine import get_engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    logger.info("🚀 تهيئة Sentiment Arabia Platform...")
    init_db()
    seed_demo_data()
    try:
        _engine = get_engine()
        logger.info("✅ محرك التحليل جاهز")
    except Exception as e:
        logger.warning(f"⚠️ محرك التحليل: {e}")
    yield
    logger.info("👋 إيقاف المنصة")

app = FastAPI(
    title="Sentiment Arabia API",
    description="منصة تحليل المشاعر العربية متعددة المستأجرين",
    version="4.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=500)

# ── WebSocket Manager ──────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, List[WebSocket]] = {}

    async def connect(self, ws: WebSocket, company_id: str):
        await ws.accept()
        if company_id not in self.active:
            self.active[company_id] = []
        self.active[company_id].append(ws)

    def disconnect(self, ws: WebSocket, company_id: str):
        if company_id in self.active:
            self.active[company_id] = [w for w in self.active[company_id] if w != ws]

    async def broadcast(self, company_id: str, data: dict):
        if company_id in self.active:
            dead = []
            for ws in self.active[company_id]:
                try:
                    await ws.send_json(data)
                except:
                    dead.append(ws)
            for ws in dead:
                self.disconnect(ws, company_id)

ws_manager = ConnectionManager()

# ══════════════════════════════════════════════════════════════════════
# Schemas (Pydantic)
# ══════════════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    id:          str   = Field(..., min_length=2, max_length=30, pattern=r'^[a-z0-9_]+$')
    name:        str   = Field(..., min_length=2, max_length=100)
    industry:    str   = Field(..., min_length=2)
    password:    str   = Field(..., min_length=6)
    email:       str   = Field(...)
    plan:        str   = Field("free")
    website:     Optional[str] = ""
    description: Optional[str] = ""
    full_name:   Optional[str] = ""

class AnalyzeRequest(BaseModel):
    text:        str   = Field(..., min_length=2, max_length=2000)
    product_id:  Optional[int]  = None
    source:      Optional[str]  = "api"

    @field_validator('text')
    @classmethod
    def clean_text(cls, v):
        return v.strip()

class BatchAnalyzeRequest(BaseModel):
    texts:      List[str] = Field(..., max_length=50)
    product_id: Optional[int] = None
    source:     Optional[str] = "batch"

class ProductCreateRequest(BaseModel):
    name:          str   = Field(..., min_length=2)
    category:      str   = Field("general")
    price:         float = Field(0, ge=0)
    currency:      str   = Field("SAR")
    description:   Optional[str] = ""
    brand:         Optional[str] = ""
    sku:           Optional[str] = ""
    availability:  int   = Field(1)
    stock_count:   int   = Field(0)
    shipping_days: int   = Field(3)
    shipping_cost: float = Field(0)
    image_url:     Optional[str] = ""
    product_url:   Optional[str] = ""
    tags:          Optional[List[str]] = []
    specs:         Optional[Dict[str, str]] = {}

class SearchRequest(BaseModel):
    query:     str   = Field("", max_length=200)
    category:  str   = Field("all")
    min_price: float = Field(0, ge=0)
    max_price: float = Field(999999, ge=0)
    sort_by:   str   = Field("relevance")
    limit:     int   = Field(20, ge=1, le=100)
    offset:    int   = Field(0, ge=0)

class WebhookRequest(BaseModel):
    url:    str
    events: Optional[List[str]] = ["comment.created", "review.submitted"]

class CompareRequest(BaseModel):
    product_ids: List[int] = Field(..., min_length=2, max_length=5)

# ══════════════════════════════════════════════════════════════════════
# Helper: Fraud Detection
# ══════════════════════════════════════════════════════════════════════

def detect_fraud(text: str, company_id: str, product_id: Optional[int] = None) -> Dict:
    flags = []
    score = 0.0

    # نمط 1: النص القصير جداً
    if len(text) < 10:
        flags.append("نص قصير جداً")
        score += 0.3

    # نمط 2: تكرار مفرط للكلمات
    words = text.split()
    if len(words) > 3:
        unique = len(set(words))
        ratio = unique / len(words)
        if ratio < 0.4:
            flags.append("تكرار مفرط للكلمات")
            score += 0.4

    # نمط 3: كلمات إغراء مشبوهة
    spam_patterns = ["اشتر الآن", "عرض لا يفوتك", "سعر خيالي", "اضغط هنا",
                     "مجاناً مجاناً", "ربح سريع", "تجاهل كل", "الأفضل في العالم"]
    for pat in spam_patterns:
        if pat in text:
            flags.append(f"عبارة ترويجية مشبوهة: {pat}")
            score += 0.25

    # نمط 4: نص عشوائي (حروف متناثرة)
    if re.search(r'[a-zA-Z]{15,}', text):
        flags.append("نص عشوائي محتمل")
        score += 0.2

    # نمط 5: مدح مبالغ فيه بدون تفاصيل
    overpraise = ["أفضل منتج في التاريخ", "لا يوجد مثيل", "الأفضل في الكون",
                  "لن تندم أبداً أبداً"]
    for pat in overpraise:
        if pat in text:
            flags.append("مدح مبالغ فيه")
            score += 0.35

    score = min(score, 1.0)
    risk_level = "عالي" if score > 0.7 else ("متوسط" if score > 0.4 else "منخفض")
    return {
        "fraud_score": round(score, 3),
        "fraud_flags": flags,
        "risk_level": risk_level,
        "is_suspicious": score > 0.5
    }

# ══════════════════════════════════════════════════════════════════════
# Helper: Extract Aspects
# ══════════════════════════════════════════════════════════════════════

def extract_aspects(text: str, sentiment: str) -> Dict[str, str]:
    aspect_keywords = {
        "جودة":    ["جودة", "متين", "متانة", "قوي", "مريح", "ممتاز", "رديء", "سيء"],
        "سعر":     ["سعر", "ثمن", "غالي", "رخيص", "تكلفة", "يستحق", "مناسب", "مرتفع"],
        "شحن":     ["شحن", "توصيل", "سريع", "تأخر", "وصل", "التسليم", "يوم", "أسبوع"],
        "خدمة":    ["خدمة", "دعم", "استجابة", "مساعدة", "فريق", "موظف", "تواصل"],
        "كاميرا":  ["كاميرا", "صورة", "تصوير", "فيديو", "دقة"],
        "بطارية":  ["بطارية", "شحن", "دوام", "يدوم", "تنتهي"],
        "شاشة":    ["شاشة", "عرض", "دقة", "وضوح", "إضاءة"],
        "أداء":    ["أداء", "سرعة", "معالج", "سلاسة", "إطارات"],
        "تغليف":   ["تغليف", "صندوق", "عبوة", "مجعد", "تالف"],
        "أصالة":   ["أصلي", "مقلد", "تقليد", "نسخة", "وهمي"],
    }
    found = {}
    for aspect, keywords in aspect_keywords.items():
        for kw in keywords:
            if kw in text:
                found[aspect] = sentiment
                break
    return found

# ══════════════════════════════════════════════════════════════════════
# ROUTES - Auth
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/v1/auth/login")
async def login(req: LoginRequest):
    user = authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="بيانات الدخول غير صحيحة")
    token = create_token({
        "company_id": user["company_id"],
        "email": user["email"],
        "role": user["role"],
        "company_name": user.get("company_name", ""),
        "plan": user.get("plan", "free"),
    })
    company = get_company(user["company_id"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "company": {
            "id": company["id"],
            "name": company["name"],
            "plan": company["plan"],
            "api_key": company["api_key"],
        }
    }

@app.post("/api/v1/auth/register")
async def register(req: RegisterRequest):
    existing = get_company(req.id)
    if existing:
        raise HTTPException(status_code=400, detail="معرف الشركة مستخدم مسبقاً")
    company = create_company(req.model_dump())
    token = create_token({
        "company_id": company["id"],
        "email": req.email,
        "role": "admin",
        "company_name": company["name"],
        "plan": company["plan"],
    })
    return {"access_token": token, "token_type": "bearer", "company": company}

@app.get("/api/v1/auth/me")
async def get_me(current: Dict = Depends(get_current_company)):
    company = get_company(current["company_id"])
    if not company:
        raise HTTPException(status_code=404, detail="الشركة غير موجودة")
    return {**current, "company": company}

# ══════════════════════════════════════════════════════════════════════
# ROUTES - Analysis
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/v1/analyze")
async def analyze(
    req: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    current: Dict = Depends(get_current_company)
):
    start = time.time()
    engine = get_engine()
    result = engine.predict(req.text)
    elapsed = time.time() - start

    fraud_info = detect_fraud(req.text, current["company_id"], req.product_id)
    aspects = extract_aspects(req.text, result.get("sentiment", "neutral"))

    analysis_data = {
        "company_id":     current["company_id"],
        "product_id":     req.product_id,
        "text":           req.text,
        "sentiment":      result.get("sentiment", "neutral"),
        "confidence":     result.get("confidence", 0),
        "positive_score": result.get("probabilities", {}).get("positive", 0),
        "negative_score": result.get("probabilities", {}).get("negative", 0),
        "neutral_score":  result.get("probabilities", {}).get("neutral", 0),
        "method":         result.get("method", "hybrid"),
        "aspects":        aspects,
        "fraud_score":    fraud_info["fraud_score"],
        "fraud_flags":    fraud_info["fraud_flags"],
        "source":         req.source or "api",
        "processing_time": elapsed,
    }
    analysis_id = save_analysis(analysis_data)

    label_map = {"positive": "إيجابي", "negative": "سلبي", "neutral": "محايد"}
    return {
        "id":             analysis_id,
        "text":           req.text,
        "sentiment":      result.get("sentiment", "neutral"),
        "sentiment_ar":   label_map.get(result.get("sentiment","neutral"), "محايد"),
        "confidence":     round(result.get("confidence", 0), 4),
        "probabilities":  result.get("probabilities", {}),
        "aspects":        aspects,
        "fraud":          fraud_info,
        "method":         result.get("method", "hybrid"),
        "processing_ms":  round(elapsed * 1000, 1),
    }

@app.post("/api/v1/analyze/batch")
async def analyze_batch(
    req: BatchAnalyzeRequest,
    current: Dict = Depends(get_current_company)
):
    engine = get_engine()
    results = []
    label_map = {"positive": "إيجابي", "negative": "سلبي", "neutral": "محايد"}
    for text in req.texts[:50]:
        start = time.time()
        r = engine.predict(text)
        elapsed = time.time() - start
        fraud_info = detect_fraud(text, current["company_id"])
        aspects = extract_aspects(text, r.get("sentiment", "neutral"))
        analysis_id = save_analysis({
            "company_id": current["company_id"],
            "product_id": req.product_id,
            "text": text,
            "sentiment": r.get("sentiment", "neutral"),
            "confidence": r.get("confidence", 0),
            "positive_score": r.get("probabilities", {}).get("positive", 0),
            "negative_score": r.get("probabilities", {}).get("negative", 0),
            "neutral_score":  r.get("probabilities", {}).get("neutral", 0),
            "method": r.get("method", "hybrid"),
            "aspects": aspects,
            "fraud_score": fraud_info["fraud_score"],
            "fraud_flags": fraud_info["fraud_flags"],
            "source": req.source or "batch",
            "processing_time": elapsed,
        })
        results.append({
            "id": analysis_id,
            "text": text[:80] + ("..." if len(text) > 80 else ""),
            "sentiment":    r.get("sentiment", "neutral"),
            "sentiment_ar": label_map.get(r.get("sentiment","neutral"), "محايد"),
            "confidence":   round(r.get("confidence", 0), 4),
            "fraud_score":  fraud_info["fraud_score"],
            "aspects":      aspects,
        })
    return {"results": results, "total": len(results)}

@app.post("/api/v1/analyze/public")
async def analyze_public(req: AnalyzeRequest):
    """تحليل عام بدون تسجيل دخول"""
    engine = get_engine()
    result = engine.predict(req.text)
    label_map = {"positive": "إيجابي", "negative": "سلبي", "neutral": "محايد"}
    fraud_info = detect_fraud(req.text, "public")
    aspects = extract_aspects(req.text, result.get("sentiment", "neutral"))
    return {
        "text":          req.text,
        "sentiment":     result.get("sentiment", "neutral"),
        "sentiment_ar":  label_map.get(result.get("sentiment","neutral"), "محايد"),
        "confidence":    round(result.get("confidence", 0), 4),
        "probabilities": result.get("probabilities", {}),
        "aspects":       aspects,
        "fraud":         fraud_info,
        "method":        result.get("method", "hybrid"),
    }

# ══════════════════════════════════════════════════════════════════════
# ROUTES - Products
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/v1/products/search")
async def product_search(req: SearchRequest):
    """بحث المنتجات عبر جميع الشركات"""
    products = search_products(
        query=req.query, category=req.category,
        min_price=req.min_price, max_price=req.max_price,
        sort_by=req.sort_by, limit=req.limit, offset=req.offset
    )
    total = len(products)
    enriched = []
    for p in products:
        stats = get_product_sentiment_stats(p["id"])
        enriched.append({
            "id": p["id"],
            "name": p["name"],
            "category": p["category"],
            "price": p["price"],
            "currency": p["currency"],
            "brand": p["brand"],
            "availability": bool(p["availability"]),
            "shipping_days": p["shipping_days"],
            "shipping_cost": p["shipping_cost"],
            "company_name": p["company_name"],
            "sentiment_score": round(p["sentiment_score"], 3),
            "positive_pct": round(p["positive_pct"], 1),
            "negative_pct": round(p["negative_pct"], 1),
            "neutral_pct": round(p["neutral_pct"], 1),
            "review_count": p["review_count"],
            "rating": p["rating"],
            "recommendation_score": round(p["recommendation_score"], 1),
            "fraud_risk": round(p["fraud_risk"], 3),
            "image_url": p.get("image_url", ""),
            "tags": p.get("tags", []),
        })
    return {"products": enriched, "total": total, "query": req.query}

@app.get("/api/v1/products/{product_id}")
async def get_product_detail(product_id: int):
    product = get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="المنتج غير موجود")
    analyses = get_product_analyses(product_id, limit=20)
    stats = get_product_sentiment_stats(product_id)
    return {
        "product": product,
        "stats": stats,
        "recent_analyses": analyses[:10]
    }

@app.get("/api/v1/products/{product_id}/compare")
async def compare_product_versions(product_id: int):
    """مقارنة المنتج عبر جميع الشركات"""
    base = get_product(product_id)
    if not base:
        raise HTTPException(status_code=404, detail="المنتج غير موجود")

    # البحث عن نفس المنتج في شركات أخرى (نفس الاسم)
    all_versions = search_products(
        query=base["name"].split()[0],
        limit=20
    )
    versions = []
    for p in all_versions:
        stats = get_product_sentiment_stats(p["id"])
        rec_score = _compute_rec_score(p, stats)
        versions.append({
            "id": p["id"],
            "company_name": p["company_name"],
            "price": p["price"],
            "availability": bool(p["availability"]),
            "shipping_days": p["shipping_days"],
            "shipping_cost": p["shipping_cost"],
            "sentiment_score": round(p["sentiment_score"], 3),
            "positive_pct": round(p["positive_pct"], 1),
            "negative_pct": round(p["negative_pct"], 1),
            "review_count": p["review_count"],
            "rating": p["rating"],
            "fraud_risk": round(p["fraud_risk"], 3),
            "recommendation_score": round(rec_score, 1),
        })

    # ترتيب وتحديد الأفضل
    versions.sort(key=lambda x: x["recommendation_score"], reverse=True)
    best = versions[0] if versions else None

    return {
        "product_name": base["name"],
        "versions": versions,
        "best_choice": best,
        "recommendation": _generate_recommendation_text(best) if best else None,
    }

@app.post("/api/v1/products/compare/multi")
async def multi_compare(req: CompareRequest):
    """مقارنة متقدمة: حتى 5 منتجات"""
    products = get_products_for_compare(req.product_ids)
    if len(products) < 2:
        raise HTTPException(status_code=400, detail="يجب اختيار منتجين على الأقل")

    comparison = []
    for p in products:
        stats = get_product_sentiment_stats(p["id"])
        rec_score = _compute_rec_score(p, stats)
        comparison.append({
            "id": p["id"],
            "name": p["name"],
            "company_name": p["company_name"],
            "category": p["category"],
            "price": p["price"],
            "currency": p["currency"],
            "availability": bool(p["availability"]),
            "shipping_days": p["shipping_days"],
            "shipping_cost": p["shipping_cost"],
            "sentiment_score": round(p["sentiment_score"], 3),
            "positive_pct": round(p["positive_pct"], 1),
            "negative_pct": round(p["negative_pct"], 1),
            "review_count": p["review_count"] or stats.get("total", 0),
            "rating": p["rating"],
            "fraud_risk": round(p["fraud_risk"], 3),
            "recommendation_score": round(rec_score, 1),
            "image_url": p.get("image_url", ""),
            "specs": p.get("specs", {}),
            "tags": p.get("tags", []),
        })

    comparison.sort(key=lambda x: x["recommendation_score"], reverse=True)
    return {
        "comparison": comparison,
        "winner": comparison[0] if comparison else None,
        "criteria": {
            "sentiment": "40%",
            "price": "30%",
            "availability": "20%",
            "fraud_risk": "10%"
        }
    }

def _compute_rec_score(product: Dict, stats: Dict) -> float:
    pos_pct = product.get("positive_pct", 0) or 0
    neg_pct = product.get("negative_pct", 0) or 0
    conf = stats.get("avg_confidence", 0.5) or 0.5
    price = product.get("price", 500) or 500
    avail = product.get("availability", 0)
    fraud = product.get("fraud_risk", 0) or 0

    sentiment_score = pos_pct * 0.40
    price_score = max(0, 100 - (price / 500) * 20) * 0.30
    avail_score = (100 if avail else 30) * 0.20
    fraud_penalty = (1 - fraud) * 100 * 0.10
    return min(100, sentiment_score + price_score + avail_score + fraud_penalty)

def _generate_recommendation_text(best: Dict) -> str:
    score = best.get("recommendation_score", 0)
    if score >= 80:
        return f"✅ {best['company_name']} هو الخيار الأفضل بدرجة توصية {score:.0f}% نظراً للسعر التنافسي والتقييمات الممتازة"
    elif score >= 60:
        return f"👍 {best['company_name']} خيار جيد بدرجة {score:.0f}%، راجع المراجعات قبل الشراء"
    else:
        return f"⚠️ جميع الخيارات لها تقييمات متوسطة، أنصح بالبحث عن منتجات بديلة"

# ══════════════════════════════════════════════════════════════════════
# ROUTES - Company Products Management
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/company/products")
async def company_products(current: Dict = Depends(get_current_company)):
    products = get_company_products(current["company_id"])
    return {"products": products, "total": len(products)}

@app.post("/api/v1/company/products")
async def add_company_product(
    req: ProductCreateRequest,
    current: Dict = Depends(get_current_company)
):
    data = req.model_dump()
    data["company_id"] = current["company_id"]
    product = create_product(data)
    return product

@app.get("/api/v1/company/analyses")
async def company_analyses(
    limit: int = 50,
    sentiment: Optional[str] = None,
    current: Dict = Depends(get_current_company)
):
    analyses = get_company_analyses(current["company_id"], limit=limit, sentiment_filter=sentiment)
    return {"analyses": analyses, "total": len(analyses)}

# ══════════════════════════════════════════════════════════════════════
# ROUTES - Dashboard
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/company/dashboard")
async def company_dashboard(current: Dict = Depends(get_current_company)):
    stats = get_company_dashboard_stats(current["company_id"])
    fraud_alerts = get_fraud_alerts(current["company_id"], limit=10)
    company = get_company(current["company_id"])
    return {
        **stats,
        "fraud_alerts": fraud_alerts,
        "company": {
            "id": company["id"],
            "name": company["name"],
            "plan": company["plan"],
            "api_key": company["api_key"],
            "webhook_secret": company.get("webhook_secret", ""),
        },
        "market_comparison": {
            "message": "أداء شركتك أفضل من 65% من الشركات في نفس المجال"
        }
    }

@app.get("/api/v1/market/insights")
async def market_insights():
    """رؤى السوق العامة (بدون كشف هوية الشركات)"""
    return get_market_insights()

# ══════════════════════════════════════════════════════════════════════
# ROUTES - Webhooks
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/v1/webhooks/config")
async def configure_webhook(
    req: WebhookRequest,
    current: Dict = Depends(get_current_company)
):
    config = create_webhook(current["company_id"], req.url, req.events)
    return config

@app.get("/api/v1/webhooks/config")
async def get_webhook(current: Dict = Depends(get_current_company)):
    config = get_webhook_config(current["company_id"])
    if not config:
        return {"message": "لا توجد إعدادات webhook", "configured": False}
    return {**config, "configured": True}

@app.post("/api/v1/webhooks/receive/{company_id}")
async def receive_webhook(
    company_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    x_signature: Optional[str] = Header(None, alias="X-Signature"),
    x_event: Optional[str] = Header(None, alias="X-Event"),
):
    """استقبال بيانات Webhook من الشركات الخارجية"""
    company = get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="الشركة غير موجودة")

    body = await request.body()

    # التحقق من التوقيع
    webhook_cfg = get_webhook_config(company_id)
    if webhook_cfg and x_signature:
        secret = company.get("webhook_secret", "")
        expected = "sha256=" + hmac.new(
            secret.encode(), body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, x_signature):
            raise HTTPException(status_code=401, detail="توقيع غير صالح")

    try:
        payload = json.loads(body)
    except:
        raise HTTPException(status_code=400, detail="بيانات JSON غير صالحة")

    event_type = x_event or payload.get("event", "comment.created")
    supported_events = ["comment.created", "review.submitted",
                        "product.rated", "order.completed"]
    if event_type not in supported_events:
        return {"status": "ignored", "event": event_type}

    # معالجة في الخلفية
    background_tasks.add_task(
        _process_webhook_event, company_id, event_type, payload
    )

    record_webhook_call(company_id, True)
    return {
        "status": "accepted",
        "event": event_type,
        "company_id": company_id,
        "timestamp": datetime.utcnow().isoformat()
    }

async def _process_webhook_event(company_id: str, event_type: str, payload: Dict):
    """معالجة حدث Webhook في الخلفية"""
    try:
        text = payload.get("text") or payload.get("comment") or payload.get("review", "")
        product_id = payload.get("product_id")
        if not text:
            return

        engine = get_engine()
        result = engine.predict(text)
        fraud_info = detect_fraud(text, company_id, product_id)
        aspects = extract_aspects(text, result.get("sentiment", "neutral"))

        analysis_id = save_analysis({
            "company_id": company_id,
            "product_id": product_id,
            "text": text,
            "sentiment": result.get("sentiment", "neutral"),
            "confidence": result.get("confidence", 0),
            "positive_score": result.get("probabilities", {}).get("positive", 0),
            "negative_score": result.get("probabilities", {}).get("negative", 0),
            "neutral_score":  result.get("probabilities", {}).get("neutral", 0),
            "method": result.get("method", "hybrid"),
            "aspects": aspects,
            "fraud_score": fraud_info["fraud_score"],
            "fraud_flags": fraud_info["fraud_flags"],
            "source": f"webhook:{event_type}",
        })

        # إرسال تحديث real-time عبر WebSocket
        await ws_manager.broadcast(company_id, {
            "type": "new_analysis",
            "event": event_type,
            "analysis_id": analysis_id,
            "sentiment": result.get("sentiment", "neutral"),
            "confidence": result.get("confidence", 0),
            "fraud_score": fraud_info["fraud_score"],
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"خطأ في معالجة webhook: {e}")

# ══════════════════════════════════════════════════════════════════════
# ROUTES - WebSocket
# ══════════════════════════════════════════════════════════════════════

@app.websocket("/ws/{company_id}")
async def websocket_endpoint(websocket: WebSocket, company_id: str):
    await ws_manager.connect(websocket, company_id)
    try:
        while True:
            data = await websocket.receive_text()
            # ping/pong للحفاظ على الاتصال
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, company_id)

# ══════════════════════════════════════════════════════════════════════
# ROUTES - Health & Info
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/health")
async def health():
    engine = get_engine()
    stats = engine.get_stats() if hasattr(engine, 'get_stats') else {}
    return {
        "status": "healthy",
        "version": "4.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "engine": stats,
        "database": "sqlite",
        "features": ["sentiment", "fraud_detection", "aspect_extraction",
                     "multi_tenant", "webhooks", "websocket", "comparison"]
    }

@app.get("/api/v1/stats")
async def global_stats():
    conn = get_db()
    c = conn.execute("SELECT COUNT(*) FROM companies WHERE is_active=1").fetchone()[0]
    p = conn.execute("SELECT COUNT(*) FROM products WHERE is_active=1").fetchone()[0]
    a = conn.execute("SELECT COUNT(*) FROM sentiment_analyses").fetchone()[0]
    pos = conn.execute("SELECT COUNT(*) FROM sentiment_analyses WHERE sentiment='positive'").fetchone()[0]
    neg = conn.execute("SELECT COUNT(*) FROM sentiment_analyses WHERE sentiment='negative'").fetchone()[0]
    conn.close()
    return {
        "companies": c,
        "products": p,
        "total_analyses": a,
        "sentiment_distribution": {
            "positive": pos,
            "negative": neg,
            "neutral": a - pos - neg
        }
    }

@app.get("/api/v1/categories")
async def get_categories():
    conn = get_db()
    rows = conn.execute("""
        SELECT category, COUNT(*) as count
        FROM products WHERE is_active=1
        GROUP BY category ORDER BY count DESC
    """).fetchall()
    conn.close()
    return {"categories": [dict(r) for r in rows]}

# ══════════════════════════════════════════════════════════════════════
# Static Files
# ══════════════════════════════════════════════════════════════════════

frontend_path = Path(__file__).parent.parent / "frontend" / "public"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index = frontend_path / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Sentiment Arabia</h1>")

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    f = frontend_path / "dashboard.html"
    if f.exists():
        return HTMLResponse(content=f.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard</h1>")

@app.get("/compare", response_class=HTMLResponse)
async def serve_compare():
    f = frontend_path / "compare.html"
    if f.exists():
        return HTMLResponse(content=f.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Compare</h1>")

@app.get("/alerts", response_class=HTMLResponse)
async def serve_alerts():
    f = frontend_path / "alerts.html"
    if f.exists():
        return HTMLResponse(content=f.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Alerts</h1>")

@app.get("/login", response_class=HTMLResponse)
async def serve_login():
    f = frontend_path / "login.html"
    if f.exists():
        return HTMLResponse(content=f.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Login</h1>")

@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(status_code=404, content={"error": "المسار غير موجود"})

@app.exception_handler(500)
async def server_error(request: Request, exc):
    logger.error(f"خطأ داخلي: {exc}")
    return JSONResponse(status_code=500, content={"error": "خطأ داخلي في الخادم"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
