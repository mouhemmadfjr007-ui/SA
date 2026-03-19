"""
Sentiment Arabia - Database Layer (SQLite Multi-Tenant)
قاعدة بيانات كاملة لمنصة تحليل المشاعر متعددة المستأجرين
"""
import sqlite3
import json
import hashlib
import secrets
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "sentiment_arabia.db"

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    """تهيئة قاعدة البيانات وإنشاء الجداول"""
    conn = get_db()
    c = conn.cursor()

    # ── جدول الشركات ──────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS companies (
        id            TEXT PRIMARY KEY,
        name          TEXT NOT NULL,
        industry      TEXT NOT NULL,
        plan          TEXT DEFAULT 'free',
        api_key       TEXT UNIQUE NOT NULL,
        webhook_secret TEXT,
        website       TEXT DEFAULT '',
        description   TEXT DEFAULT '',
        logo_url      TEXT DEFAULT '',
        is_active     INTEGER DEFAULT 1,
        analyses_count INTEGER DEFAULT 0,
        monthly_limit  INTEGER DEFAULT 1000,
        created_at    TEXT DEFAULT (datetime('now')),
        updated_at    TEXT DEFAULT (datetime('now'))
    )""")

    # ── جدول مستخدمي الشركات ──────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS company_users (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id  TEXT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
        email       TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role        TEXT DEFAULT 'admin',
        full_name   TEXT DEFAULT '',
        is_active   INTEGER DEFAULT 1,
        last_login  TEXT,
        created_at  TEXT DEFAULT (datetime('now'))
    )""")

    # ── جدول المنتجات ─────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id      TEXT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
        name            TEXT NOT NULL,
        name_en         TEXT DEFAULT '',
        category        TEXT DEFAULT 'general',
        price           REAL DEFAULT 0,
        currency        TEXT DEFAULT 'SAR',
        description     TEXT DEFAULT '',
        image_url       TEXT DEFAULT '',
        product_url     TEXT DEFAULT '',
        sku             TEXT DEFAULT '',
        brand           TEXT DEFAULT '',
        availability    INTEGER DEFAULT 1,
        stock_count     INTEGER DEFAULT 0,
        shipping_days   INTEGER DEFAULT 3,
        shipping_cost   REAL DEFAULT 0,
        rating          REAL DEFAULT 0,
        review_count    INTEGER DEFAULT 0,
        sentiment_score REAL DEFAULT 0,
        positive_pct    REAL DEFAULT 0,
        negative_pct    REAL DEFAULT 0,
        neutral_pct     REAL DEFAULT 0,
        fraud_risk      REAL DEFAULT 0,
        recommendation_score REAL DEFAULT 0,
        tags            TEXT DEFAULT '[]',
        specs           TEXT DEFAULT '{}',
        is_active       INTEGER DEFAULT 1,
        created_at      TEXT DEFAULT (datetime('now')),
        updated_at      TEXT DEFAULT (datetime('now'))
    )""")

    # ── جدول تحليلات المشاعر ──────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS sentiment_analyses (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id      TEXT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
        product_id      INTEGER REFERENCES products(id) ON DELETE SET NULL,
        text            TEXT NOT NULL,
        sentiment       TEXT NOT NULL,
        confidence      REAL DEFAULT 0,
        positive_score  REAL DEFAULT 0,
        negative_score  REAL DEFAULT 0,
        neutral_score   REAL DEFAULT 0,
        method          TEXT DEFAULT 'hybrid',
        aspects         TEXT DEFAULT '{}',
        fraud_score     REAL DEFAULT 0,
        fraud_flags     TEXT DEFAULT '[]',
        source          TEXT DEFAULT 'api',
        language        TEXT DEFAULT 'ar',
        processing_time REAL DEFAULT 0,
        created_at      TEXT DEFAULT (datetime('now'))
    )""")

    # ── جدول إعدادات Webhooks ─────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS webhooks_config (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id  TEXT UNIQUE NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
        url         TEXT NOT NULL,
        secret      TEXT NOT NULL,
        events      TEXT DEFAULT '["comment.created","review.submitted"]',
        is_active   INTEGER DEFAULT 1,
        retry_count INTEGER DEFAULT 3,
        timeout_sec INTEGER DEFAULT 30,
        last_called TEXT,
        total_calls INTEGER DEFAULT 0,
        failed_calls INTEGER DEFAULT 0,
        created_at  TEXT DEFAULT (datetime('now'))
    )""")

    # ── جدول مقارنات المنتجات ─────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS product_comparisons (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id      TEXT NOT NULL,
        product_ids     TEXT NOT NULL,
        winner_id       INTEGER REFERENCES products(id),
        comparison_data TEXT DEFAULT '{}',
        user_ip         TEXT DEFAULT '',
        created_at      TEXT DEFAULT (datetime('now'))
    )""")

    # ── جدول تنبيهات الاحتيال ─────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS fraud_alerts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id  TEXT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
        product_id  INTEGER REFERENCES products(id),
        analysis_id INTEGER REFERENCES sentiment_analyses(id),
        alert_type  TEXT NOT NULL,
        risk_level  TEXT NOT NULL,
        description TEXT DEFAULT '',
        details     TEXT DEFAULT '{}',
        is_resolved INTEGER DEFAULT 0,
        resolved_at TEXT,
        created_at  TEXT DEFAULT (datetime('now'))
    )""")

    # ── جدول منتجات المستخدمين المحفوظة ──────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS user_saved_products (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
        notes      TEXT DEFAULT '',
        created_at TEXT DEFAULT (datetime('now')),
        UNIQUE(session_id, product_id)
    )""")

    # ── الفهارس للأداء ────────────────────────────────────────────────
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_products_company ON products(company_id)",
        "CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)",
        "CREATE INDEX IF NOT EXISTS idx_products_name ON products(name)",
        "CREATE INDEX IF NOT EXISTS idx_products_sentiment ON products(sentiment_score)",
        "CREATE INDEX IF NOT EXISTS idx_products_price ON products(price)",
        "CREATE INDEX IF NOT EXISTS idx_analyses_company ON sentiment_analyses(company_id)",
        "CREATE INDEX IF NOT EXISTS idx_analyses_product ON sentiment_analyses(product_id)",
        "CREATE INDEX IF NOT EXISTS idx_analyses_sentiment ON sentiment_analyses(sentiment)",
        "CREATE INDEX IF NOT EXISTS idx_analyses_created ON sentiment_analyses(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_fraud_company ON fraud_alerts(company_id)",
        "CREATE INDEX IF NOT EXISTS idx_fraud_risk ON fraud_alerts(risk_level)",
    ]
    for idx in indexes:
        c.execute(idx)

    conn.commit()
    conn.close()

# ══════════════════════════════════════════════════════════════════════
# CRUD - Companies
# ══════════════════════════════════════════════════════════════════════

def hash_password(password: str) -> str:
    try:
        import bcrypt
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    except:
        return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    try:
        import bcrypt
        if hashed.startswith('$2'):
            return bcrypt.checkpw(password.encode(), hashed.encode())
    except:
        pass
    return hashlib.sha256(password.encode()).hexdigest() == hashed

def generate_api_key() -> str:
    return "sa_" + secrets.token_hex(24)

def get_company(company_id: str) -> Optional[Dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM companies WHERE id=?", (company_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_companies() -> List[Dict]:
    conn = get_db()
    rows = conn.execute("SELECT * FROM companies WHERE is_active=1").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def authenticate_user(email: str, password: str) -> Optional[Dict]:
    conn = get_db()
    user = conn.execute(
        "SELECT u.*, c.name as company_name, c.plan, c.is_active as company_active "
        "FROM company_users u JOIN companies c ON u.company_id=c.id "
        "WHERE u.email=? AND u.is_active=1", (email,)
    ).fetchone()
    if not user:
        conn.close()
        return None
    if not verify_password(password, user['password_hash']):
        conn.close()
        return None
    conn.execute("UPDATE company_users SET last_login=datetime('now') WHERE id=?", (user['id'],))
    conn.commit()
    conn.close()
    return dict(user)

def create_company(data: Dict) -> Dict:
    conn = get_db()
    api_key = generate_api_key()
    webhook_secret = secrets.token_hex(16)
    plan_limits = {'free': 1000, 'basic': 10000, 'premium': 100000, 'enterprise': 9999999}
    monthly_limit = plan_limits.get(data.get('plan', 'free'), 1000)

    conn.execute("""
        INSERT INTO companies (id, name, industry, plan, api_key, webhook_secret,
                               website, description, monthly_limit)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (data['id'], data['name'], data['industry'],
          data.get('plan', 'free'), api_key, webhook_secret,
          data.get('website', ''), data.get('description', ''), monthly_limit))

    conn.execute("""
        INSERT INTO company_users (company_id, email, password_hash, role, full_name)
        VALUES (?,?,?,?,?)
    """, (data['id'], data['email'], hash_password(data['password']),
          'admin', data.get('full_name', data['name'])))

    conn.commit()
    conn.close()
    return get_company(data['id'])

def get_company_by_api_key(api_key: str) -> Optional[Dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM companies WHERE api_key=? AND is_active=1", (api_key,)).fetchone()
    conn.close()
    return dict(row) if row else None

# ══════════════════════════════════════════════════════════════════════
# CRUD - Products
# ══════════════════════════════════════════════════════════════════════

def get_product(product_id: int) -> Optional[Dict]:
    conn = get_db()
    row = conn.execute("""
        SELECT p.*, c.name as company_name, c.industry as company_industry
        FROM products p JOIN companies c ON p.company_id=c.id
        WHERE p.id=? AND p.is_active=1
    """, (product_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d['tags'] = json.loads(d.get('tags') or '[]')
    d['specs'] = json.loads(d.get('specs') or '{}')
    return d

def search_products(query: str = "", category: str = "", min_price: float = 0,
                    max_price: float = 999999, sort_by: str = "relevance",
                    limit: int = 20, offset: int = 0) -> List[Dict]:
    conn = get_db()
    sql = """
        SELECT p.*, c.name as company_name, c.industry as company_industry
        FROM products p JOIN companies c ON p.company_id=c.id
        WHERE p.is_active=1 AND c.is_active=1
    """
    params = []
    if query:
        sql += " AND (p.name LIKE ? OR p.description LIKE ? OR p.brand LIKE ?)"
        q = f"%{query}%"
        params += [q, q, q]
    if category and category != "all":
        sql += " AND p.category=?"
        params.append(category)
    if min_price > 0:
        sql += " AND p.price >= ?"
        params.append(min_price)
    if max_price < 999999:
        sql += " AND p.price <= ?"
        params.append(max_price)

    sort_map = {
        "relevance": "p.recommendation_score DESC",
        "price_asc": "p.price ASC",
        "price_desc": "p.price DESC",
        "rating": "p.sentiment_score DESC",
        "reviews": "p.review_count DESC",
        "newest": "p.created_at DESC"
    }
    sql += f" ORDER BY {sort_map.get(sort_by, 'p.recommendation_score DESC')}"
    sql += " LIMIT ? OFFSET ?"
    params += [limit, offset]

    rows = conn.execute(sql, params).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d['tags'] = json.loads(d.get('tags') or '[]')
        d['specs'] = json.loads(d.get('specs') or '{}')
        result.append(d)
    return result

def get_company_products(company_id: str, limit: int = 50) -> List[Dict]:
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM products WHERE company_id=? AND is_active=1
        ORDER BY recommendation_score DESC LIMIT ?
    """, (company_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def create_product(data: Dict) -> Dict:
    conn = get_db()
    score = _compute_recommendation_score(
        data.get('sentiment_score', 0.5),
        data.get('price', 100),
        data.get('availability', 1),
        data.get('fraud_risk', 0),
    )
    c = conn.cursor()
    c.execute("""
        INSERT INTO products (company_id, name, name_en, category, price, currency,
            description, image_url, product_url, sku, brand, availability,
            stock_count, shipping_days, shipping_cost, rating, review_count,
            sentiment_score, positive_pct, negative_pct, neutral_pct,
            fraud_risk, recommendation_score, tags, specs)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data['company_id'], data['name'], data.get('name_en', ''),
        data.get('category', 'general'), data.get('price', 0),
        data.get('currency', 'SAR'), data.get('description', ''),
        data.get('image_url', ''), data.get('product_url', ''),
        data.get('sku', ''), data.get('brand', ''),
        data.get('availability', 1), data.get('stock_count', 0),
        data.get('shipping_days', 3), data.get('shipping_cost', 0),
        data.get('rating', 0), data.get('review_count', 0),
        data.get('sentiment_score', 0.5),
        data.get('positive_pct', 50), data.get('negative_pct', 25),
        data.get('neutral_pct', 25), data.get('fraud_risk', 0), score,
        json.dumps(data.get('tags', []), ensure_ascii=False),
        json.dumps(data.get('specs', {}), ensure_ascii=False)
    ))
    product_id = c.lastrowid
    conn.commit()
    conn.close()
    return get_product(product_id)

def _compute_recommendation_score(sentiment: float, price: float,
                                   availability: int, fraud_risk: float) -> float:
    """خوارزمية التوصية: وزن مركّب"""
    sentiment_score = min(sentiment * 100, 100) * 0.40
    price_score = max(0, 100 - (price / 200) * 20) * 0.30
    avail_score = (100 if availability else 30) * 0.20
    fraud_score = max(0, 100 - fraud_risk * 100) * 0.10
    return round(sentiment_score + price_score + avail_score + fraud_score, 2)

def update_product_stats(product_id: int):
    """تحديث إحصائيات المنتج من التحليلات"""
    conn = get_db()
    stats = conn.execute("""
        SELECT
            COUNT(*) as total,
            AVG(confidence) as avg_conf,
            SUM(CASE WHEN sentiment='positive' THEN 1 ELSE 0 END) as pos,
            SUM(CASE WHEN sentiment='negative' THEN 1 ELSE 0 END) as neg,
            SUM(CASE WHEN sentiment='neutral'  THEN 1 ELSE 0 END) as neu,
            AVG(fraud_score) as avg_fraud
        FROM sentiment_analyses
        WHERE product_id=?
    """, (product_id,)).fetchone()
    if stats and stats['total'] > 0:
        total = stats['total']
        pos_pct = (stats['pos'] / total) * 100
        neg_pct = (stats['neg'] / total) * 100
        neu_pct = (stats['neu'] / total) * 100
        sentiment_score = (pos_pct - neg_pct + 100) / 200
        fraud_risk = stats['avg_fraud'] or 0
        rec_score = _compute_recommendation_score(
            sentiment_score, 0, 1, fraud_risk
        )
        conn.execute("""
            UPDATE products SET
                review_count=?, positive_pct=?, negative_pct=?, neutral_pct=?,
                sentiment_score=?, fraud_risk=?, recommendation_score=?,
                updated_at=datetime('now')
            WHERE id=?
        """, (total, pos_pct, neg_pct, neu_pct, sentiment_score,
              fraud_risk, rec_score, product_id))
        conn.commit()
    conn.close()

# ══════════════════════════════════════════════════════════════════════
# CRUD - Analyses
# ══════════════════════════════════════════════════════════════════════

def save_analysis(data: Dict) -> int:
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO sentiment_analyses
            (company_id, product_id, text, sentiment, confidence,
             positive_score, negative_score, neutral_score,
             method, aspects, fraud_score, fraud_flags, source, processing_time)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data['company_id'], data.get('product_id'),
        data['text'], data['sentiment'], data.get('confidence', 0),
        data.get('positive_score', 0), data.get('negative_score', 0),
        data.get('neutral_score', 0), data.get('method', 'hybrid'),
        json.dumps(data.get('aspects', {}), ensure_ascii=False),
        data.get('fraud_score', 0),
        json.dumps(data.get('fraud_flags', []), ensure_ascii=False),
        data.get('source', 'api'), data.get('processing_time', 0)
    ))
    analysis_id = c.lastrowid

    # تحديث عداد الشركة
    conn.execute("UPDATE companies SET analyses_count=analyses_count+1 WHERE id=?",
                 (data['company_id'],))

    # تحديث إحصائيات المنتج
    if data.get('product_id'):
        conn.commit()
        conn.close()
        update_product_stats(data['product_id'])
        return analysis_id

    # كشف الاحتيال تلقائياً
    if data.get('fraud_score', 0) > 0.7:
        conn.execute("""
            INSERT INTO fraud_alerts (company_id, product_id, analysis_id,
                alert_type, risk_level, description, details)
            VALUES (?,?,?,?,?,?,?)
        """, (
            data['company_id'], data.get('product_id'), analysis_id,
            'high_fraud_score', 'high' if data['fraud_score'] > 0.85 else 'medium',
            f"نص بدرجة احتيال مرتفعة: {data['fraud_score']:.2f}",
            json.dumps({'fraud_flags': data.get('fraud_flags', [])}, ensure_ascii=False)
        ))

    conn.commit()
    conn.close()
    return analysis_id

def get_product_analyses(product_id: int, limit: int = 50) -> List[Dict]:
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM sentiment_analyses
        WHERE product_id=? ORDER BY created_at DESC LIMIT ?
    """, (product_id, limit)).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d['aspects'] = json.loads(d.get('aspects') or '{}')
        d['fraud_flags'] = json.loads(d.get('fraud_flags') or '[]')
        result.append(d)
    return result

def get_company_analyses(company_id: str, limit: int = 100,
                          sentiment_filter: str = None) -> List[Dict]:
    conn = get_db()
    sql = "SELECT * FROM sentiment_analyses WHERE company_id=?"
    params = [company_id]
    if sentiment_filter:
        sql += " AND sentiment=?"
        params.append(sentiment_filter)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_product_sentiment_stats(product_id: int) -> Dict:
    conn = get_db()
    stats = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN sentiment='positive' THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN sentiment='negative' THEN 1 ELSE 0 END) as negative,
            SUM(CASE WHEN sentiment='neutral'  THEN 1 ELSE 0 END) as neutral,
            AVG(confidence) as avg_confidence,
            AVG(fraud_score) as avg_fraud,
            MAX(created_at) as last_analysis
        FROM sentiment_analyses WHERE product_id=?
    """, (product_id,)).fetchone()
    conn.close()
    if not stats or not stats['total']:
        return {'total': 0, 'positive': 0, 'negative': 0, 'neutral': 0,
                'avg_confidence': 0, 'avg_fraud': 0}
    return dict(stats)

# ══════════════════════════════════════════════════════════════════════
# Dashboard & Market Insights
# ══════════════════════════════════════════════════════════════════════

def get_company_dashboard_stats(company_id: str) -> Dict:
    conn = get_db()

    # إجماليات
    totals = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN sentiment='positive' THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN sentiment='negative' THEN 1 ELSE 0 END) as negative,
            SUM(CASE WHEN sentiment='neutral'  THEN 1 ELSE 0 END) as neutral,
            AVG(confidence) as avg_confidence
        FROM sentiment_analyses WHERE company_id=?
    """, (company_id,)).fetchone()

    # أفضل المنتجات
    top_products = conn.execute("""
        SELECT name, recommendation_score, positive_pct, review_count
        FROM products WHERE company_id=? AND is_active=1
        ORDER BY recommendation_score DESC LIMIT 5
    """, (company_id,)).fetchall()

    # أسوأ المنتجات
    worst_products = conn.execute("""
        SELECT name, recommendation_score, negative_pct, review_count
        FROM products WHERE company_id=? AND is_active=1 AND review_count > 0
        ORDER BY negative_pct DESC LIMIT 5
    """, (company_id,)).fetchall()

    # تنبيهات الاحتيال
    fraud_alerts = conn.execute("""
        SELECT COUNT(*) as count FROM fraud_alerts
        WHERE company_id=? AND is_resolved=0
    """, (company_id,)).fetchone()

    # اتجاهات المشاعر (7 أيام)
    trends = conn.execute("""
        SELECT
            DATE(created_at) as day,
            COUNT(*) as total,
            SUM(CASE WHEN sentiment='positive' THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN sentiment='negative' THEN 1 ELSE 0 END) as negative
        FROM sentiment_analyses
        WHERE company_id=? AND created_at >= datetime('now', '-7 days')
        GROUP BY DATE(created_at)
        ORDER BY day ASC
    """, (company_id,)).fetchall()

    # الجوانب الأكثر ذكراً
    aspects_raw = conn.execute("""
        SELECT aspects FROM sentiment_analyses
        WHERE company_id=? AND aspects != '{}' LIMIT 200
    """, (company_id,)).fetchall()
    aspect_counts = {}
    for row in aspects_raw:
        try:
            asp = json.loads(row['aspects'])
            for k, v in asp.items():
                if v:
                    aspect_counts[k] = aspect_counts.get(k, 0) + 1
        except:
            pass

    # معلومات الشركة والحصة
    company = conn.execute(
        "SELECT * FROM companies WHERE id=?", (company_id,)
    ).fetchone()

    conn.close()
    t = dict(totals) if totals else {}
    total = t.get('total', 0) or 0
    positive = t.get('positive', 0) or 0
    negative = t.get('negative', 0) or 0
    neutral = t.get('neutral', 0) or 0

    return {
        "totals": {
            "analyses": total,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "avg_confidence": round(t.get('avg_confidence', 0) or 0, 3),
            "positive_pct": round(positive / max(total, 1) * 100, 1),
            "negative_pct": round(negative / max(total, 1) * 100, 1),
            "neutral_pct":  round(neutral  / max(total, 1) * 100, 1),
        },
        "top_products": [dict(r) for r in top_products],
        "worst_products": [dict(r) for r in worst_products],
        "fraud_alerts_count": (dict(fraud_alerts)['count'] if fraud_alerts else 0),
        "trends": [dict(r) for r in trends],
        "top_aspects": sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        "plan": dict(company)['plan'] if company else 'free',
        "usage": {
            "used": dict(company)['analyses_count'] if company else 0,
            "limit": dict(company)['monthly_limit'] if company else 1000
        }
    }

def get_market_insights() -> Dict:
    conn = get_db()

    # متوسط المشاعر
    market_avg = conn.execute("""
        SELECT
            AVG(CASE WHEN sentiment='positive' THEN 1.0 ELSE 0 END) * 100 as avg_positive,
            AVG(CASE WHEN sentiment='negative' THEN 1.0 ELSE 0 END) * 100 as avg_negative,
            COUNT(*) as total_analyses,
            AVG(confidence) as avg_confidence
        FROM sentiment_analyses
    """).fetchone()

    # المنتجات الرائجة (أعلى توصية)
    trending = conn.execute("""
        SELECT p.name, p.category, p.recommendation_score, p.positive_pct,
               p.review_count, c.industry
        FROM products p JOIN companies c ON p.company_id=c.id
        WHERE p.is_active=1 AND p.review_count > 2
        ORDER BY p.recommendation_score DESC LIMIT 10
    """).fetchall()

    # توزيع الفئات
    categories = conn.execute("""
        SELECT category, COUNT(*) as count, AVG(recommendation_score) as avg_score
        FROM products WHERE is_active=1
        GROUP BY category ORDER BY count DESC
    """).fetchall()

    # نطاقات الأسعار
    price_ranges = conn.execute("""
        SELECT
            CASE
                WHEN price < 100 THEN 'أقل من 100 ريال'
                WHEN price < 500 THEN '100-500 ريال'
                WHEN price < 1000 THEN '500-1000 ريال'
                ELSE 'أكثر من 1000 ريال'
            END as range,
            COUNT(*) as count,
            AVG(sentiment_score) as avg_sentiment
        FROM products WHERE is_active=1
        GROUP BY range ORDER BY count DESC
    """).fetchall()

    # بؤر الاحتيال
    fraud_hotspots = conn.execute("""
        SELECT alert_type, risk_level, COUNT(*) as count
        FROM fraud_alerts
        GROUP BY alert_type, risk_level
        ORDER BY count DESC LIMIT 5
    """).fetchall()

    conn.close()
    m = dict(market_avg) if market_avg else {}
    return {
        "market_average": {
            "positive_pct": round(m.get('avg_positive', 0) or 0, 1),
            "negative_pct": round(m.get('avg_negative', 0) or 0, 1),
            "total_analyses": m.get('total_analyses', 0),
            "avg_confidence": round(m.get('avg_confidence', 0) or 0, 3),
        },
        "trending_products": [dict(r) for r in trending],
        "categories": [dict(r) for r in categories],
        "price_ranges": [dict(r) for r in price_ranges],
        "fraud_hotspots": [dict(r) for r in fraud_hotspots],
    }

# ══════════════════════════════════════════════════════════════════════
# Webhooks
# ══════════════════════════════════════════════════════════════════════

def create_webhook(company_id: str, url: str, events: List[str] = None) -> Dict:
    conn = get_db()
    secret = secrets.token_hex(20)
    events_json = json.dumps(events or ["comment.created", "review.submitted"])
    conn.execute("""
        INSERT OR REPLACE INTO webhooks_config
            (company_id, url, secret, events)
        VALUES (?,?,?,?)
    """, (company_id, url, secret, events_json))
    conn.commit()
    conn.close()
    return get_webhook_config(company_id)

def get_webhook_config(company_id: str) -> Optional[Dict]:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM webhooks_config WHERE company_id=?", (company_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d['events'] = json.loads(d.get('events') or '[]')
    return d

def record_webhook_call(company_id: str, success: bool):
    conn = get_db()
    if success:
        conn.execute("""
            UPDATE webhooks_config SET total_calls=total_calls+1,
            last_called=datetime('now') WHERE company_id=?
        """, (company_id,))
    else:
        conn.execute("""
            UPDATE webhooks_config SET total_calls=total_calls+1,
            failed_calls=failed_calls+1, last_called=datetime('now')
            WHERE company_id=?
        """, (company_id,))
    conn.commit()
    conn.close()

# ══════════════════════════════════════════════════════════════════════
# Compare Products (Multi-Company)
# ══════════════════════════════════════════════════════════════════════

def get_products_for_compare(product_ids: List[int]) -> List[Dict]:
    if not product_ids:
        return []
    conn = get_db()
    placeholders = ",".join("?" * len(product_ids))
    rows = conn.execute(f"""
        SELECT p.*, c.name as company_name, c.industry
        FROM products p JOIN companies c ON p.company_id=c.id
        WHERE p.id IN ({placeholders}) AND p.is_active=1
    """, product_ids).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d['tags'] = json.loads(d.get('tags') or '[]')
        d['specs'] = json.loads(d.get('specs') or '{}')
        result.append(d)
    return result

def get_fraud_alerts(company_id: str, limit: int = 20) -> List[Dict]:
    conn = get_db()
    rows = conn.execute("""
        SELECT fa.*, p.name as product_name
        FROM fraud_alerts fa LEFT JOIN products p ON fa.product_id=p.id
        WHERE fa.company_id=? ORDER BY fa.created_at DESC LIMIT ?
    """, (company_id, limit)).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d['details'] = json.loads(d.get('details') or '{}')
        except:
            d['details'] = {}
        result.append(d)
    return result

def seed_demo_data():
    """بيانات تجريبية افتراضية"""
    conn = get_db()
    existing = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    conn.close()
    if existing > 0:
        return

    companies = [
        {"id": "noon_sa",  "name": "نون السعودية",    "industry": "تجارة إلكترونية", "plan": "premium",
         "email": "admin@noon_sa.com",    "password": "noon2024",   "website": "https://noon.com/saudi-ar/"},
        {"id": "amazon_sa","name": "أمازون السعودية", "industry": "تجارة إلكترونية", "plan": "enterprise",
         "email": "admin@amazon_sa.com",  "password": "amazon2024", "website": "https://amazon.sa"},
        {"id": "jarir",    "name": "جرير",            "industry": "إلكترونيات", "plan": "basic",
         "email": "admin@jarir.com",      "password": "jarir2024",  "website": "https://jarir.com"},
        {"id": "extra_sa", "name": "إكسترا",          "industry": "إلكترونيات", "plan": "basic",
         "email": "admin@extra_sa.com",   "password": "extra2024",  "website": "https://extra.com"},
        {"id": "saco_sa",  "name": "ساكو",            "industry": "أدوات منزلية", "plan": "free",
         "email": "admin@saco_sa.com",    "password": "saco2024",   "website": "https://saco.com.sa"},
    ]
    for c in companies:
        create_company(c)

    products_data = [
        # نون
        {"company_id": "noon_sa", "name": "سامسونج جالكسي S24 Ultra",
         "category": "هواتف ذكية", "price": 4999, "brand": "Samsung",
         "availability": 1, "stock_count": 45, "shipping_days": 1, "shipping_cost": 0,
         "sentiment_score": 0.82, "positive_pct": 82, "negative_pct": 8, "neutral_pct": 10,
         "review_count": 127, "rating": 4.5,
         "description": "هاتف رائد بكاميرا 200 ميجابيكسل وشاشة Dynamic AMOLED",
         "tags": ["هاتف", "سامسونج", "أندرويد"],
         "specs": {"الشاشة": "6.8 بوصة", "الذاكرة": "12GB RAM", "التخزين": "256GB"}},
        {"company_id": "noon_sa", "name": "آيفون 15 برو ماكس",
         "category": "هواتف ذكية", "price": 5999, "brand": "Apple",
         "availability": 1, "stock_count": 30, "shipping_days": 1, "shipping_cost": 0,
         "sentiment_score": 0.88, "positive_pct": 88, "negative_pct": 5, "neutral_pct": 7,
         "review_count": 203, "rating": 4.8,
         "description": "أقوى آيفون بشريحة A17 Pro وإمكانيات تصوير احترافية",
         "tags": ["آيفون", "أبل", "iOS"],
         "specs": {"الشاشة": "6.7 بوصة", "الذاكرة": "8GB RAM", "التخزين": "256GB"}},
        {"company_id": "noon_sa", "name": "لابتوب ديل XPS 15",
         "category": "حاسبات", "price": 6800, "brand": "Dell",
         "availability": 1, "stock_count": 12, "shipping_days": 2, "shipping_cost": 0,
         "sentiment_score": 0.75, "positive_pct": 75, "negative_pct": 12, "neutral_pct": 13,
         "review_count": 89, "rating": 4.2,
         "description": "لابتوب احترافي بمعالج Intel Core i7 وشاشة OLED",
         "tags": ["لابتوب", "ديل", "ويندوز"],
         "specs": {"المعالج": "Intel i7-13700H", "الذاكرة": "16GB", "التخزين": "512GB SSD"}},
        {"company_id": "noon_sa", "name": "سماعات سوني WH-1000XM5",
         "category": "إلكترونيات", "price": 1299, "brand": "Sony",
         "availability": 1, "stock_count": 67, "shipping_days": 1, "shipping_cost": 0,
         "sentiment_score": 0.91, "positive_pct": 91, "negative_pct": 3, "neutral_pct": 6,
         "review_count": 156, "rating": 4.9,
         "description": "أفضل سماعات لإلغاء الضوضاء في السوق",
         "tags": ["سماعات", "سوني", "بلوتوث"],
         "specs": {"مدة البطارية": "30 ساعة", "إلغاء الضوضاء": "نعم", "الاتصال": "بلوتوث 5.2"}},
        {"company_id": "noon_sa", "name": "تلفاز LG OLED 65 بوصة",
         "category": "تلفزيونات", "price": 8500, "brand": "LG",
         "availability": 0, "stock_count": 0, "shipping_days": 5, "shipping_cost": 150,
         "sentiment_score": 0.79, "positive_pct": 79, "negative_pct": 10, "neutral_pct": 11,
         "review_count": 44, "rating": 4.4,
         "description": "تلفاز OLED بدقة 4K وتقنية صوت Dolby Atmos",
         "tags": ["تلفاز", "LG", "OLED"],
         "specs": {"الدقة": "4K", "التقنية": "OLED", "HDR": "Dolby Vision"}},
        # أمازون
        {"company_id": "amazon_sa", "name": "سامسونج جالكسي S24 Ultra",
         "category": "هواتف ذكية", "price": 4799, "brand": "Samsung",
         "availability": 1, "stock_count": 120, "shipping_days": 1, "shipping_cost": 0,
         "sentiment_score": 0.84, "positive_pct": 84, "negative_pct": 7, "neutral_pct": 9,
         "review_count": 312, "rating": 4.6,
         "description": "جالكسي S24 Ultra مع قلم S Pen وكاميرا احترافية",
         "tags": ["هاتف", "سامسونج"],
         "specs": {"الشاشة": "6.8 بوصة", "الذاكرة": "12GB RAM", "التخزين": "256GB"}},
        {"company_id": "amazon_sa", "name": "آيفون 15 برو ماكس",
         "category": "هواتف ذكية", "price": 5799, "brand": "Apple",
         "availability": 1, "stock_count": 80, "shipping_days": 1, "shipping_cost": 0,
         "sentiment_score": 0.90, "positive_pct": 90, "negative_pct": 4, "neutral_pct": 6,
         "review_count": 445, "rating": 4.9,
         "description": "آيفون 15 برو ماكس مع إمكانية التصوير بدقة 4K ProRes",
         "tags": ["آيفون", "أبل"],
         "specs": {"الشاشة": "6.7 بوصة", "الذاكرة": "8GB RAM", "التخزين": "256GB"}},
        {"company_id": "amazon_sa", "name": "كيندل باب ربيك 11",
         "category": "أجهزة القراءة", "price": 499, "brand": "Amazon",
         "availability": 1, "stock_count": 200, "shipping_days": 2, "shipping_cost": 0,
         "sentiment_score": 0.87, "positive_pct": 87, "negative_pct": 5, "neutral_pct": 8,
         "review_count": 88, "rating": 4.7,
         "description": "قارئ كتب إلكترونية خفيف الوزن بشاشة عالية الدقة",
         "tags": ["قراءة", "أمازون", "كتب"],
         "specs": {"الشاشة": "6 بوصة", "التخزين": "16GB", "البطارية": "أسابيع"}},
        # جرير
        {"company_id": "jarir", "name": "لابتوب ماك بوك برو M3",
         "category": "حاسبات", "price": 9500, "brand": "Apple",
         "availability": 1, "stock_count": 25, "shipping_days": 2, "shipping_cost": 0,
         "sentiment_score": 0.93, "positive_pct": 93, "negative_pct": 3, "neutral_pct": 4,
         "review_count": 198, "rating": 4.9,
         "description": "ماك بوك برو بشريحة M3 Pro الأقوى لمحترفي الإبداع",
         "tags": ["ماك", "أبل", "لابتوب"],
         "specs": {"المعالج": "Apple M3 Pro", "الذاكرة": "18GB", "التخزين": "512GB SSD"}},
        {"company_id": "jarir", "name": "سامسونج جالكسي S24 Ultra",
         "category": "هواتف ذكية", "price": 5200, "brand": "Samsung",
         "availability": 1, "stock_count": 18, "shipping_days": 3, "shipping_cost": 25,
         "sentiment_score": 0.80, "positive_pct": 80, "negative_pct": 10, "neutral_pct": 10,
         "review_count": 67, "rating": 4.3,
         "description": "جالكسي S24 Ultra بضمان جرير الذهبي",
         "tags": ["هاتف", "سامسونج"],
         "specs": {"الشاشة": "6.8 بوصة", "الذاكرة": "12GB RAM", "التخزين": "256GB"}},
        {"company_id": "jarir", "name": "طابعة HP LaserJet Pro",
         "category": "طابعات", "price": 1200, "brand": "HP",
         "availability": 1, "stock_count": 40, "shipping_days": 3, "shipping_cost": 50,
         "sentiment_score": 0.72, "positive_pct": 72, "negative_pct": 14, "neutral_pct": 14,
         "review_count": 53, "rating": 4.1,
         "description": "طابعة ليزر عالية السرعة مثالية للمكاتب",
         "tags": ["طابعة", "HP", "ليزر"],
         "specs": {"السرعة": "30 ص/دقيقة", "الدقة": "1200 DPI", "الاتصال": "WiFi"}},
        # إكسترا
        {"company_id": "extra_sa", "name": "ثلاجة سامسونج 650 لتر",
         "category": "أجهزة منزلية", "price": 3800, "brand": "Samsung",
         "availability": 1, "stock_count": 15, "shipping_days": 4, "shipping_cost": 0,
         "sentiment_score": 0.76, "positive_pct": 76, "negative_pct": 13, "neutral_pct": 11,
         "review_count": 42, "rating": 4.2,
         "description": "ثلاجة فرنش دور بتقنية No Frost وشاشة ذكية",
         "tags": ["ثلاجة", "سامسونج", "منزلي"],
         "specs": {"السعة": "650 لتر", "التقنية": "No Frost", "الطاقة": "A++"}},
        {"company_id": "extra_sa", "name": "مكيف ميديا 24000 وحدة",
         "category": "أجهزة منزلية", "price": 2200, "brand": "Midea",
         "availability": 1, "stock_count": 30, "shipping_days": 7, "shipping_cost": 0,
         "sentiment_score": 0.68, "positive_pct": 68, "negative_pct": 18, "neutral_pct": 14,
         "review_count": 95, "rating": 3.9,
         "description": "مكيف سبليت بطاقة 24000 BTU وخاصية التبريد السريع",
         "tags": ["مكيف", "ميديا"],
         "specs": {"الطاقة": "24000 BTU", "كفاءة الطاقة": "A+", "نوع": "سبليت"}},
        {"company_id": "extra_sa", "name": "آيفون 15 برو ماكس",
         "category": "هواتف ذكية", "price": 6200, "brand": "Apple",
         "availability": 0, "stock_count": 0, "shipping_days": 7, "shipping_cost": 0,
         "sentiment_score": 0.85, "positive_pct": 85, "negative_pct": 8, "neutral_pct": 7,
         "review_count": 29, "rating": 4.6,
         "description": "آيفون 15 برو ماكس - متاح للطلب المسبق",
         "tags": ["آيفون", "أبل"],
         "specs": {"الشاشة": "6.7 بوصة", "التخزين": "256GB"}},
        # ساكو
        {"company_id": "saco_sa", "name": "مضخة ضغط كارشر K5",
         "category": "أدوات", "price": 750, "brand": "Kärcher",
         "availability": 1, "stock_count": 22, "shipping_days": 3, "shipping_cost": 45,
         "sentiment_score": 0.83, "positive_pct": 83, "negative_pct": 7, "neutral_pct": 10,
         "review_count": 38, "rating": 4.5,
         "description": "مضخة ضغط عالية القوة لتنظيف السيارات والأسطح الخارجية",
         "tags": ["أدوات", "تنظيف", "كارشر"],
         "specs": {"الضغط": "145 بار", "الطاقة": "2100 وات", "التدفق": "500 لتر/ساعة"}},
        {"company_id": "saco_sa", "name": "طقم أدوات بوش 120 قطعة",
         "category": "أدوات", "price": 950, "brand": "Bosch",
         "availability": 1, "stock_count": 10, "shipping_days": 3, "shipping_cost": 45,
         "sentiment_score": 0.78, "positive_pct": 78, "negative_pct": 10, "neutral_pct": 12,
         "review_count": 24, "rating": 4.4,
         "description": "طقم أدوات احترافي شامل لجميع أعمال الإصلاح",
         "tags": ["أدوات", "بوش", "إصلاح"],
         "specs": {"عدد القطع": "120", "المواد": "فولاذ مصلد", "الصندوق": "بلاستيك متين"}},
    ]
    for p in products_data:
        create_product(p)

    # إضافة تحليلات نموذجية
    sample_reviews = [
        # سامسونج جالكسي - إيجابية
        ("noon_sa",  1, "الهاتف رائع جداً والكاميرا احترافية بشكل لا يصدق، أنصح به بشدة",  "positive", 0.92),
        ("noon_sa",  1, "أفضل هاتف اشتريته في حياتي، البطارية تدوم يومين كاملين",           "positive", 0.88),
        ("noon_sa",  1, "الشاشة ممتازة والأداء سلس جداً لكن السعر مرتفع نسبياً",           "neutral",  0.75),
        ("noon_sa",  1, "وصل مكسور! الشاشة بها خطوط والصندوق تالف. خدمة سيئة",            "negative", 0.91),
        ("amazon_sa",6, "أفضل سعر وجدته للجالكسي S24، الشحن كان سريع جداً",               "positive", 0.87),
        ("amazon_sa",6, "المنتج أصلي 100% والسعر تنافسي، التوصيل في اليوم التالي",          "positive", 0.90),
        ("amazon_sa",6, "جودة المنتج ممتازة لكن لا يوجد ضمان محلي",                        "neutral",  0.72),
        ("jarir",   10, "اشتريته من جرير بضمان 2 سنة، الجودة عالية لكن السعر أغلى",        "neutral",  0.68),
        ("jarir",   10, "الضمان من جرير يستحق الفرق في السعر، خدمة ما بعد البيع ممتازة",   "positive", 0.85),
        # آيفون - إيجابية
        ("noon_sa",  2, "آيفون 15 برو ماكس تحفة فنية، الكاميرا والأداء لا يوصفان",        "positive", 0.94),
        ("noon_sa",  2, "اشتريته لزوجتي وهي سعيدة جداً، الجهاز خفيف وسريع",               "positive", 0.89),
        ("amazon_sa",7, "أسرع تسليم وأقل سعر للآيفون 15 برو ماكس في السعودية",            "positive", 0.93),
        ("amazon_sa",7, "الجهاز ممتاز والشحن في نفس اليوم، أمازون لا تخذل",               "positive", 0.91),
        ("extra_sa",14, "توقعت أقل من هذا السعر بكثير! أمازون ونون أوفر بكثير",           "negative", 0.83),
        # سوني سماعات
        ("noon_sa",  4, "أفضل سماعات جربتها في حياتي، إلغاء الضوضاء يعمل بشكل سحري",     "positive", 0.96),
        ("noon_sa",  4, "الصوت نقي جداً والراحة ممتازة عند الاستخدام لساعات طويلة",        "positive", 0.92),
        ("noon_sa",  4, "مريحة جداً وعزل الصوت ممتاز لكن يتعب الأذن بعد 4 ساعات",        "neutral",  0.78),
        ("noon_sa",  4, "استلمتها ومكسورة! لا أنصح بالشراء من هذا البائع",                "negative", 0.88),
        # ماك بوك جرير
        ("jarir",    9, "الماك بوك برو M3 أسرع جهاز استخدمته، يستحق كل ريال",             "positive", 0.95),
        ("jarir",    9, "ضمان جرير الذهبي يجعل الفرق في السعر يستحق التفكير",              "positive", 0.82),
        ("jarir",    9, "الجهاز رائع لكن السعر أغلى من الأسواق الأخرى بـ 500 ريال",       "neutral",  0.73),
        # مكيف إكسترا
        ("extra_sa",13,"المكيف يبرد بسرعة لكن الصوت عالٍ نوعاً ما في الليل",              "neutral",  0.71),
        ("extra_sa",13,"بعد 3 أشهر توقف عن العمل! الضمان رفض الإصلاح المجاني",           "negative", 0.89),
        ("extra_sa",13,"التركيب مجاني والمكيف ممتاز للغرف الكبيرة",                       "positive", 0.81),
        # أدوات ساكو
        ("saco_sa", 15,"مضخة الكارشر قوية جداً، تنظف السيارة في 10 دقائق",               "positive", 0.88),
        ("saco_sa", 15,"جودة ممتازة والضغط قوي لكن الخرطوم قصير نسبياً",                "neutral",  0.74),
        ("saco_sa", 16,"طقم بوش يستحق السعر، الأدوات متينة ومريحة الإمساك",              "positive", 0.86),
    ]

    for company_id, product_id, text, sentiment, confidence in sample_reviews:
        score_map = {"positive": (confidence, 0.05, 0.05),
                     "negative": (0.05, confidence, 0.05),
                     "neutral":  (0.15, 0.15, confidence)}
        pos, neg, neu = score_map[sentiment]
        save_analysis({
            "company_id": company_id, "product_id": product_id,
            "text": text, "sentiment": sentiment, "confidence": confidence,
            "positive_score": pos, "negative_score": neg, "neutral_score": neu,
            "method": "seed", "aspects": {}, "fraud_score": 0.02,
            "source": "seed"
        })

    print("✅ تم إدراج البيانات التجريبية بنجاح")
    print("   شركات: 5 | منتجات: 16 | مراجعات: 26")

# تشغيل التهيئة
if __name__ == "__main__":
    init_db()
    seed_demo_data()
