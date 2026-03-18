"""
Sentiment Arabia - FastAPI Backend v2
خادم REST API كامل ومحسّن لتحليل المشاعر العربية
"""
import os, sys, json, time, logging, asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from collections import defaultdict

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.inference.optimized_inference_engine import (
    get_engine, SentimentInferenceEngine, detect_fraud, extract_aspects
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger(__name__)

# ─── Pydantic Models ──────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="النص المراد تحليله")
    company_id:  Optional[str] = Field(None, description="معرّف الشركة")
    product_id:  Optional[str] = Field(None, description="معرّف المنتج")
    include_aspects: bool = Field(False, description="تضمين استخراج الجوانب")
    include_fraud:   bool = Field(False, description="تضمين كشف الاحتيال")

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v):
        if not v.strip(): raise ValueError('النص لا يمكن أن يكون فارغاً')
        return v.strip()


class BatchAnalyzeRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)
    company_id:      Optional[str] = None
    include_aspects: bool = False
    include_fraud:   bool = False


class CompanyCreateRequest(BaseModel):
    id:       str = Field(..., min_length=1, max_length=50)
    name:     str = Field(..., min_length=1, max_length=200)
    industry: Optional[str] = None
    products: Optional[List[str]] = []


class ProductSearchRequest(BaseModel):
    query:      str = Field(..., min_length=1)
    company_id: Optional[str] = None
    limit:      int = Field(10, ge=1, le=50)


# ─── In-Memory Stores ─────────────────────────────────────────────────
analysis_history: List[dict] = []
companies_db: Dict[str, dict] = {
    'demo_co': {'id': 'demo_co', 'name': 'شركة تجريبية', 'industry': 'تقنية',
                'products': ['منتج أ', 'منتج ب'], 'created_at': datetime.now().isoformat()}
}
products_db: Dict[str, dict] = {
    'prod_1': {'id': 'prod_1', 'name': 'منتج أ', 'company_id': 'demo_co', 'category': 'إلكترونيات'},
    'prod_2': {'id': 'prod_2', 'name': 'منتج ب', 'company_id': 'demo_co', 'category': 'ملابس'}
}
_inference_engine: Optional[SentimentInferenceEngine] = None


# ─── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _inference_engine
    logger.info("🚀 Starting Sentiment Arabia API...")
    _inference_engine = get_engine()
    logger.info(f"✅ Engine ready  model_loaded={_inference_engine.model_loaded}")
    yield
    logger.info("🛑 Shutting down...")


# ─── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Arabia API",
    description="نظام تحليل المشاعر العربية للشركات",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=500)

# Mount frontend
FRONTEND_DIR = Path(__file__).parent.parent / 'frontend' / 'public'
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def get_inference_engine() -> SentimentInferenceEngine:
    if _inference_engine is None:
        raise HTTPException(503, "نظام الاستدلال غير جاهز")
    return _inference_engine


# ─── Routes ───────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    idx = FRONTEND_DIR / 'index.html'
    if idx.exists():
        return HTMLResponse(idx.read_text(encoding='utf-8'))
    return HTMLResponse("<h1>Sentiment Arabia API - v2.0</h1><p><a href='/docs'>API Docs</a></p>")


@app.get("/api/v1/health")
async def health(engine: SentimentInferenceEngine = Depends(get_inference_engine)):
    return {
        "status": "healthy",
        "model_loaded": engine.model_loaded,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "engine_stats": engine.stats()
    }


@app.post("/api/v1/analyze")
async def analyze(
    req: AnalyzeRequest,
    bg: BackgroundTasks,
    engine: SentimentInferenceEngine = Depends(get_inference_engine)
):
    result = engine.predict(
        req.text,
        include_aspects=req.include_aspects,
        include_fraud=req.include_fraud
    )
    if 'error' in result:
        raise HTTPException(400, result['error'])

    response = {
        "success": True,
        "text": req.text[:200],
        "sentiment":    result['sentiment'],
        "sentiment_ar": result['sentiment_ar'],
        "label":        result['label'],
        "confidence":   result['confidence'],
        "probabilities": result['probabilities'],
        "method":       result['method'],
        "latency_ms":   result['latency_ms'],
        "from_cache":   result['from_cache'],
        "timestamp":    datetime.now().isoformat()
    }
    if req.include_aspects and 'aspects' in result:
        response['aspects'] = result['aspects']
    if req.include_fraud and 'fraud_detection' in result:
        response['fraud_detection'] = result['fraud_detection']

    bg.add_task(_save_history, req.text, result, req.company_id, req.product_id)
    return response


@app.post("/api/v1/analyze/batch")
async def analyze_batch(
    req: BatchAnalyzeRequest,
    engine: SentimentInferenceEngine = Depends(get_inference_engine)
):
    if len(req.texts) > 100:
        raise HTTPException(400, "الحد الأقصى 100 نص في الطلب الواحد")

    t0 = time.time()
    results = engine.predict_batch(
        req.texts,
        include_aspects=req.include_aspects,
        include_fraud=req.include_fraud
    )
    total_ms = round((time.time() - t0) * 1000, 2)

    dist = defaultdict(int)
    for r in results:
        dist[r.get('sentiment', 'unknown')] += 1

    return {
        "success": True,
        "total": len(results),
        "results": results,
        "summary": {
            "distribution": dict(dist),
            "avg_confidence": round(sum(r.get('confidence', 0) for r in results) / len(results), 4),
            "total_latency_ms": total_ms
        },
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/analyze/demo")
async def analyze_demo(engine: SentimentInferenceEngine = Depends(get_inference_engine)):
    demo_texts = [
        "المنتج ممتاز جداً وجودته عالية، سعيد جداً بشرائه 😊",
        "خدمة سيئة جداً والتوصيل تأخر كثيراً ولن أعود 😡",
        "المنتج عادي لا شيء مميز يستحق الذكر",
        "رائع! أفضل تجربة في حياتي، أنصح بشدة 🌟",
        "لا أنصح بهذا المنتج - مزيف ومقلد وخداع للمشتري",
        "التوصيل سريع والتغليف ممتاز، شكراً على الخدمة الاحترافية 👍"
    ]
    results = engine.predict_batch(demo_texts, include_aspects=True, include_fraud=True)
    return {
        "success": True,
        "demo_results": [
            {
                "text": text,
                "sentiment": r['sentiment'],
                "sentiment_ar": r['sentiment_ar'],
                "confidence": round(r['confidence'], 3),
                "method": r['method'],
                "probabilities": {k: round(v, 3) for k, v in r['probabilities'].items()},
                "aspects": r.get('aspects', []),
                "fraud": r.get('fraud_detection', {})
            }
            for text, r in zip(demo_texts, results)
        ]
    }


@app.get("/api/v1/stats")
async def stats(engine: SentimentInferenceEngine = Depends(get_inference_engine)):
    total = len(analysis_history)
    dist = defaultdict(int)
    for h in analysis_history[-1000:]:
        dist[h.get('sentiment', 'unknown')] += 1
    return {
        "success": True,
        "summary": {
            "total_analyses":  total,
            "total_companies": len(companies_db),
            "total_products":  len(products_db),
            "sentiment_distribution": dict(dist),
            "model_status": "active" if engine.model_loaded else "fallback",
            "cache_hit_rate": engine.cache.hit_rate,
            "avg_latency_ms": round(engine.avg_latency, 2)
        },
        "engine": engine.stats(),
        "timestamp": datetime.now().isoformat()
    }


# ─── Company Routes ────────────────────────────────────────────────────
@app.post("/api/v1/companies")
async def create_company(req: CompanyCreateRequest):
    if req.id in companies_db:
        raise HTTPException(409, f"الشركة '{req.id}' موجودة مسبقاً")
    companies_db[req.id] = {
        "id": req.id, "name": req.name, "industry": req.industry,
        "products": req.products, "created_at": datetime.now().isoformat()
    }
    return {"success": True, "company": companies_db[req.id]}


@app.get("/api/v1/companies")
async def list_companies():
    return {"success": True, "companies": list(companies_db.values()), "total": len(companies_db)}


@app.get("/api/v1/companies/{company_id}")
async def get_company(company_id: str):
    if company_id not in companies_db:
        raise HTTPException(404, f"الشركة '{company_id}' غير موجودة")
    co = companies_db[company_id]
    # Get recent analyses for this company
    recent = [h for h in analysis_history[-500:] if h.get('company_id') == company_id][-20:]
    dist = defaultdict(int)
    for h in recent:
        dist[h.get('sentiment', 'unknown')] += 1
    return {
        "success": True, "company": co,
        "analytics": {
            "total_analyses": len(recent),
            "sentiment_distribution": dict(dist)
        }
    }


# ─── Product Routes ────────────────────────────────────────────────────
@app.post("/api/v1/products/search")
async def search_products(req: ProductSearchRequest):
    query = req.query.lower()
    results = []
    for pid, prod in products_db.items():
        if (req.company_id and prod.get('company_id') != req.company_id):
            continue
        if query in prod.get('name', '').lower() or query in prod.get('category', '').lower():
            results.append(prod)
    return {"success": True, "results": results[:req.limit], "total": len(results)}


@app.get("/api/v1/products")
async def list_products(company_id: Optional[str] = None):
    products = list(products_db.values())
    if company_id:
        products = [p for p in products if p.get('company_id') == company_id]
    return {"success": True, "products": products, "total": len(products)}


# ─── Analysis History ─────────────────────────────────────────────────
@app.get("/api/v1/history")
async def get_history(limit: int = 50, company_id: Optional[str] = None):
    hist = analysis_history[-1000:]
    if company_id:
        hist = [h for h in hist if h.get('company_id') == company_id]
    return {
        "success": True,
        "history": hist[-limit:],
        "total":   len(hist)
    }


# ─── Background Tasks ─────────────────────────────────────────────────
async def _save_history(text: str, result: dict, company_id: str = None, product_id: str = None):
    analysis_history.append({
        'text_preview': text[:100],
        'sentiment':    result.get('sentiment'),
        'confidence':   result.get('confidence'),
        'company_id':   company_id,
        'product_id':   product_id,
        'timestamp':    datetime.now().isoformat()
    })
    if len(analysis_history) > 10000:
        analysis_history.pop(0)


# ─── Error Handlers ───────────────────────────────────────────────────
@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(404, {"success": False, "error": "المسار غير موجود", "path": str(request.url.path)})


@app.exception_handler(500)
async def server_error(request: Request, exc):
    logger.error(f"خطأ داخلي: {exc}")
    return JSONResponse(500, {"success": False, "error": "خطأ داخلي في الخادم"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info", workers=1)
