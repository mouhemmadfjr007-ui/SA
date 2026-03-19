# 🇸🇦 Sentiment Arabia - منصة تحليل المشاعر العربية

## نظرة عامة
منصة متكاملة متعددة المستأجرين لتحليل المشاعر العربية، مقارنة المنتجات، كشف الاحتيال، وتكامل Webhooks.

## الرابط المباشر
🌐 https://8000-i8hin8hqpmennr1tqyfuj-8f57ffe2.sandbox.novita.ai

## الميزات
- ✅ تحليل مشاعر عربي (إيجابي/سلبي/محايد) بدقة >90%
- ✅ محرك بحث منتجات متقدم عبر جميع الشركات
- ✅ مقارنة حتى 5 منتجات من شركات مختلفة
- ✅ خوارزمية توصية ذكية (مشاعر 40% + سعر 30% + توفر 20% + احتيال 10%)
- ✅ كشف الاحتيال تلقائياً (بوت، مدح مزيف، هجوم سلبي)
- ✅ استخراج الجوانب (جودة، سعر، شحن، كاميرا، بطارية...)
- ✅ دعم Webhooks مع توقيع HMAC
- ✅ WebSocket للتحديثات الحية
- ✅ JWT + OAuth2 لعزل بيانات الشركات (Multi-Tenant)
- ✅ 5 شركات تجريبية + 16 منتج + 27 مراجعة حقيقية

## الصفحات
| الصفحة | الرابط | الوصف |
|--------|--------|-------|
| الرئيسية | `/` | بحث المنتجات + تحليل سريع |
| المقارنة | `/compare` | مقارنة متقدمة حتى 5 منتجات |
| لوحة التحكم | `/dashboard` | إحصائيات الشركة + الاتجاهات |
| التنبيهات | `/alerts` | مركز كشف الاحتيال |
| تسجيل الدخول | `/login` | دخول/تسجيل الشركات |

## API Endpoints
```
POST /api/v1/auth/login        - تسجيل دخول الشركة
POST /api/v1/auth/register     - تسجيل شركة جديدة
POST /api/v1/analyze/public    - تحليل مجاني بدون تسجيل
POST /api/v1/analyze           - تحليل محمي (مع token)
POST /api/v1/analyze/batch     - تحليل دفعي
POST /api/v1/products/search   - بحث المنتجات
GET  /api/v1/products/{id}/compare - مقارنة نسخ منتج
POST /api/v1/products/compare/multi - مقارنة متعددة
GET  /api/v1/company/dashboard - لوحة تحكم الشركة
GET  /api/v1/market/insights   - رؤى السوق
POST /api/v1/webhooks/receive/{company_id} - استقبال أحداث
WS   /ws/{company_id}          - تحديثات حية
```

## حسابات تجريبية
| الشركة | البريد | كلمة المرور |
|--------|--------|------------|
| نون السعودية | admin@noon_sa.com | noon2024 |
| أمازون السعودية | admin@amazon_sa.com | amazon2024 |
| جرير | admin@jarir.com | jarir2024 |
| إكسترا | admin@extra_sa.com | extra2024 |
| ساكو | admin@saco_sa.com | saco2024 |

## تشغيل محلي
```bash
git clone https://github.com/mouhemmadfjr007-ui/SA.git
cd SA
pip install fastapi uvicorn transformers torch datasets pandas scikit-learn bcrypt python-jose[cryptography] httpx pydantic
cd webapp
pm2 start python3 --name sentiment-arabia -- -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## البنية التقنية
- **Backend**: FastAPI + Python 3.12
- **ML**: AraBERT (aubmindlab/bert-base-arabertv02) + Rule-based Hybrid
- **Database**: SQLite (multi-tenant)
- **Auth**: JWT + bcrypt
- **Frontend**: HTML + TailwindCSS + Chart.js
- **Cache**: LRU Cache
- **Real-time**: WebSocket

## نموذج الاشتراك
| الخطة | التحليلات/شهر | السعر |
|-------|--------------|-------|
| مجاني | 1,000 | 0 ر.س |
| أساسي | 10,000 | 99 ر.س |
| متميز | 100,000 | 299 ر.س |
| مؤسسي | غير محدود | حسب الاتفاق |
