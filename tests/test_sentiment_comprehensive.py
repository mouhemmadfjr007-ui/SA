"""
====================================================
Sentiment Arabia - Comprehensive Tests
====================================================
اختبارات شاملة تغطي:
- التحليل الأساسي
- حالات الحافة (Edge Cases)
- الأداء والسرعة
- الاتساق (Consistency)
- اختبارات التكامل
====================================================
"""

import sys
import time
import json
import logging
import requests
import statistics
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

# ============================
# Test Results Tracker
# ============================
class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.warnings = []
        self.performance_data = []
    
    def success(self, name: str, details: str = ""):
        self.passed += 1
        logger.info(f"  ✅ {name}" + (f": {details}" if details else ""))
    
    def fail(self, name: str, details: str = ""):
        self.failed += 1
        self.errors.append(f"{name}: {details}")
        logger.error(f"  ❌ {name}" + (f": {details}" if details else ""))
    
    def warn(self, name: str, details: str = ""):
        self.warnings.append(f"{name}: {details}")
        logger.warning(f"  ⚠️ {name}" + (f": {details}" if details else ""))
    
    def summary(self):
        total = self.passed + self.failed
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 نتائج الاختبارات:")
        logger.info(f"   ✅ نجح: {self.passed}/{total}")
        logger.info(f"   ❌ فشل: {self.failed}/{total}")
        if self.warnings:
            logger.info(f"   ⚠️ تحذيرات: {len(self.warnings)}")
        
        if self.performance_data:
            latencies = [p['latency'] for p in self.performance_data]
            logger.info(f"\n⚡ إحصائيات الأداء:")
            logger.info(f"   متوسط: {statistics.mean(latencies):.2f}ms")
            logger.info(f"   P95: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}ms")
            logger.info(f"   P99: {sorted(latencies)[int(len(latencies)*0.99)]:.2f}ms")
        
        if self.errors:
            logger.info(f"\n❌ الأخطاء:")
            for err in self.errors:
                logger.info(f"   - {err}")
        
        success_rate = (self.passed / total * 100) if total > 0 else 0
        logger.info(f"\n{'='*60}")
        logger.info(f"{'✅' if success_rate >= 80 else '❌'} معدل النجاح: {success_rate:.1f}%")
        
        return success_rate >= 80


results = TestResults()


# ============================
# Test Utilities
# ============================

def post(endpoint: str, data: dict, timeout: int = 30) -> dict:
    resp = requests.post(f"{BASE_URL}{endpoint}", json=data, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def get(endpoint: str, timeout: int = 30) -> dict:
    resp = requests.get(f"{BASE_URL}{endpoint}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ============================
# Tests
# ============================

def test_01_health():
    """اختبار صحة الخادم"""
    logger.info("\n--- 1. اختبار الصحة ---")
    
    try:
        resp = get("/api/v1/health")
        assert resp.get("status") in ["healthy", "degraded"], "حالة غير متوقعة"
        results.success("Health Check", f"status={resp['status']}")
    except Exception as e:
        results.fail("Health Check", str(e))


def test_02_basic_sentiment():
    """اختبار تحليل المشاعر الأساسي"""
    logger.info("\n--- 2. اختبار التحليل الأساسي ---")
    
    test_cases = [
        {
            "text": "المنتج ممتاز جداً وجودته عالية وسعيد جداً بشرائه",
            "expected": "positive",
            "description": "نص إيجابي واضح"
        },
        {
            "text": "خدمة سيئة جداً والتوصيل تأخر أسبوعاً كاملاً",
            "expected": "negative",
            "description": "نص سلبي واضح"
        },
        {
            "text": "المنتج عادي يؤدي الغرض لكن لا شيء مميز",
            "expected": "neutral",
            "description": "نص محايد"
        },
    ]
    
    for case in test_cases:
        try:
            start = time.time()
            resp = post("/api/v1/analyze", {"text": case["text"]})
            latency = (time.time() - start) * 1000
            
            assert resp.get("success"), "الطلب لم ينجح"
            result = resp.get("result", {})
            
            sentiment = result.get("sentiment")
            confidence = result.get("confidence", 0)
            
            if sentiment == case["expected"]:
                results.success(
                    case["description"],
                    f"sentiment={sentiment}, confidence={confidence:.2%}, latency={latency:.0f}ms"
                )
            else:
                results.warn(
                    case["description"],
                    f"توقعنا {case['expected']} لكن حصلنا على {sentiment} (confidence={confidence:.2%})"
                )
            
            results.performance_data.append({
                'test': case["description"],
                'latency': latency,
                'sentiment': sentiment
            })
            
        except Exception as e:
            results.fail(case["description"], str(e))


def test_03_confidence_scores():
    """اختبار درجات الثقة"""
    logger.info("\n--- 3. اختبار درجات الثقة ---")
    
    try:
        resp = post("/api/v1/analyze", {
            "text": "أفضل منتج اشتريته في حياتي، ممتاز جداً"
        })
        result = resp.get("result", {})
        
        # Check confidence range
        confidence = result.get("confidence", 0)
        assert 0 <= confidence <= 1, f"الثقة خارج النطاق: {confidence}"
        results.success("نطاق الثقة", f"{confidence:.4f}")
        
        # Check probabilities sum to ~1
        probs = result.get("probabilities", {})
        prob_sum = sum(probs.values())
        assert abs(prob_sum - 1.0) < 0.01, f"مجموع الاحتمالات خاطئ: {prob_sum}"
        results.success("مجموع الاحتمالات", f"≈ {prob_sum:.4f}")
        
        # Check all three classes present
        assert all(k in probs for k in ['positive', 'negative', 'neutral']), "فئات مفقودة"
        results.success("الفئات الثلاث موجودة")
        
    except Exception as e:
        results.fail("درجات الثقة", str(e))


def test_04_edge_cases():
    """اختبار حالات الحافة"""
    logger.info("\n--- 4. اختبار حالات الحافة ---")
    
    edge_cases = [
        {
            "text": "جيد",
            "name": "نص قصير جداً (كلمة واحدة)"
        },
        {
            "text": "المنتج ممتاز" * 50,  # Long text
            "name": "نص طويل جداً"
        },
        {
            "text": "هذا المنتج... ممتاز!!! 👍👍👍",
            "name": "نص مع رموز وتعجب"
        },
        {
            "text": "مرحبا Hello Bonjour مزيج لغات",
            "name": "نص متعدد اللغات"
        },
        {
            "text": "123456789",
            "name": "أرقام فقط"
        },
        {
            "text": "😊😊😊 رائع",
            "name": "إيموجي مع نص"
        },
    ]
    
    for case in edge_cases:
        try:
            # Truncate for display
            text_display = case["text"][:50] + "..." if len(case["text"]) > 50 else case["text"]
            
            resp = post("/api/v1/analyze", {"text": case["text"][:5000]})
            result = resp.get("result", {})
            
            assert result.get("sentiment") in ["positive", "negative", "neutral"]
            assert 0 <= result.get("confidence", 0) <= 1
            
            results.success(case["name"], f"sentiment={result['sentiment']}")
            
        except Exception as e:
            results.warn(case["name"], str(e))


def test_05_consistency():
    """اختبار الاتساق (نفس النص يعطي نفس النتيجة)"""
    logger.info("\n--- 5. اختبار الاتساق ---")
    
    test_text = "المنتج ممتاز وأنصح به بشدة"
    
    try:
        results_list = []
        for i in range(5):
            resp = post("/api/v1/analyze", {"text": test_text})
            sentiment = resp["result"]["sentiment"]
            confidence = resp["result"]["confidence"]
            results_list.append(sentiment)
        
        # All should be same
        unique = set(results_list)
        if len(unique) == 1:
            results.success("الاتساق", f"نفس النتيجة {len(results_list)} مرة: {results_list[0]}")
        else:
            results.warn("الاتساق", f"نتائج مختلفة: {unique}")
            
    except Exception as e:
        results.fail("الاتساق", str(e))


def test_06_caching():
    """اختبار نظام الكاش"""
    logger.info("\n--- 6. اختبار الكاش ---")
    
    test_text = "هذا نص اختباري للكاش - يجب أن يكون أسرع في الطلب الثاني"
    
    try:
        # First request
        start1 = time.time()
        resp1 = post("/api/v1/analyze", {"text": test_text})
        t1 = (time.time() - start1) * 1000
        
        # Second request (should be faster from cache)
        start2 = time.time()
        resp2 = post("/api/v1/analyze", {"text": test_text})
        t2 = (time.time() - start2) * 1000
        
        cached = resp2["result"].get("cached", False)
        
        if cached or t2 < t1 * 0.8:
            results.success("الكاش يعمل", f"طلب1={t1:.0f}ms, طلب2={t2:.0f}ms, cached={cached}")
        else:
            results.warn("الكاش", f"الكاش لم يكن أسرع: {t1:.0f}ms vs {t2:.0f}ms")
            
    except Exception as e:
        results.fail("الكاش", str(e))


def test_07_batch_processing():
    """اختبار المعالجة الدفعية"""
    logger.info("\n--- 7. اختبار المعالجة الدفعية ---")
    
    texts = [
        "المنتج ممتاز جداً وأنصح به",
        "خدمة سيئة جداً ولا أنصح بها",
        "المنتج عادي لا شيء مميز",
        "أفضل تجربة شراء في حياتي",
        "التوصيل تأخر كثيراً"
    ]
    
    try:
        start = time.time()
        resp = post("/api/v1/analyze/batch", {"texts": texts})
        t = (time.time() - start) * 1000
        
        assert resp.get("success"), "الطلب فشل"
        assert resp.get("total") == len(texts), "عدد النتائج خاطئ"
        
        batch_results = resp.get("results", [])
        assert len(batch_results) == len(texts), "النتائج غير مكتملة"
        
        for result in batch_results:
            assert result.get("sentiment") in ["positive", "negative", "neutral"]
        
        stats = resp.get("batch_stats", {})
        results.success(
            "المعالجة الدفعية",
            f"{len(texts)} نص في {t:.0f}ms | "
            f"إيجابي:{stats.get('sentiment_distribution',{}).get('positive',0)} | "
            f"سلبي:{stats.get('sentiment_distribution',{}).get('negative',0)}"
        )
        
    except Exception as e:
        results.fail("المعالجة الدفعية", str(e))


def test_08_performance():
    """اختبار الأداء"""
    logger.info("\n--- 8. اختبار الأداء ---")
    
    test_texts = [
        "المنتج ممتاز وجودته عالية",
        "لا أنصح بهذه الخدمة السيئة",
        "الطعام جيد لكن الخدمة بطيئة",
        "أفضل موقع للتسوق الإلكتروني",
        "المشكلة لم تحل حتى الآن",
    ]
    
    latencies = []
    
    try:
        for text in test_texts:
            start = time.time()
            resp = post("/api/v1/analyze", {"text": text})
            latency = (time.time() - start) * 1000
            latencies.append(latency)
        
        avg = statistics.mean(latencies)
        p95 = sorted(latencies)[int(len(latencies)*0.95)] if len(latencies) > 1 else latencies[0]
        
        if avg < 500:  # Under 500ms average
            results.success(
                "الأداء - متوسط الاستجابة",
                f"avg={avg:.0f}ms, p95={p95:.0f}ms"
            )
        else:
            results.warn(
                "الأداء - متوسط الاستجابة",
                f"بطيء: avg={avg:.0f}ms (الهدف < 500ms)"
            )
        
        # Add to global performance data
        for lat in latencies:
            results.performance_data.append({'test': 'performance', 'latency': lat})
            
    except Exception as e:
        results.fail("الأداء", str(e))


def test_09_companies_api():
    """اختبار API الشركات"""
    logger.info("\n--- 9. اختبار API الشركات ---")
    
    try:
        # Create company
        resp = post("/api/v1/companies", {
            "name": "شركة اختبار التقنية",
            "industry": "تقنية المعلومات",
            "description": "شركة اختبار"
        })
        
        assert resp.get("success"), "إنشاء الشركة فشل"
        company_id = resp.get("company_id")
        results.success("إنشاء شركة", f"ID={company_id}")
        
        # List companies
        list_resp = get("/api/v1/companies")
        assert list_resp.get("success"), "قائمة الشركات فشلت"
        assert list_resp.get("total", 0) > 0, "لا توجد شركات"
        results.success("قائمة الشركات", f"{list_resp['total']} شركة")
        
        # Get specific company
        detail_resp = get(f"/api/v1/companies/{company_id}")
        assert detail_resp.get("success"), "تفاصيل الشركة فشلت"
        results.success("تفاصيل الشركة", f"name={detail_resp['company']['name']}")
        
    except Exception as e:
        results.fail("API الشركات", str(e))


def test_10_product_search():
    """اختبار البحث عن المنتجات"""
    logger.info("\n--- 10. اختبار البحث عن المنتجات ---")
    
    try:
        resp = post("/api/v1/products/search", {
            "query": "هاتف",
            "limit": 5
        })
        
        assert resp.get("success"), "البحث فشل"
        products = resp.get("products", [])
        
        results.success("البحث عن منتجات", f"وجد {len(products)} منتج لـ 'هاتف'")
        
        if products:
            prod = products[0]
            has_analysis = "sentiment_analysis" in prod
            if has_analysis:
                analysis = prod['sentiment_analysis']
                results.success(
                    "تحليل مشاعر المنتج",
                    f"overall={analysis.get('overall_sentiment')}, "
                    f"positive={analysis.get('positive')}, "
                    f"negative={analysis.get('negative')}"
                )
        
    except Exception as e:
        results.fail("البحث عن المنتجات", str(e))


def test_11_arabic_specific():
    """اختبارات خاصة باللغة العربية"""
    logger.info("\n--- 11. اختبارات اللغة العربية ---")
    
    arabic_tests = [
        # Various Arabic dialects
        ("والله المنتج زين جداً يستاهل", "positive", "لهجة خليجية إيجابية"),
        ("يا ريت ما اشتريته، خراب", "negative", "نص سلبي بالعامية"),
        ("عادي مش كويس ولا وحش", "neutral", "نص محايد بالعامية المصرية"),
        
        # Modern Arabic
        ("تجربة رائعة وخدمة احترافية ممتازة", "positive", "فصحى إيجابية"),
        ("غير مرضٍ على الإطلاق", "negative", "فصحى سلبية"),
        
        # Mixed with English
        ("الـ quality ممتازة والـ delivery سريعة", "positive", "عربي-إنجليزي إيجابي"),
    ]
    
    for text, expected, description in arabic_tests:
        try:
            resp = post("/api/v1/analyze", {"text": text})
            result = resp.get("result", {})
            sentiment = result.get("sentiment")
            confidence = result.get("confidence", 0)
            
            if sentiment == expected:
                results.success(description, f"{sentiment} ({confidence:.1%})")
            else:
                results.warn(description, f"توقعنا {expected}، حصلنا {sentiment} ({confidence:.1%})")
                
        except Exception as e:
            results.warn(description, str(e))


def test_12_demo_endpoint():
    """اختبار نقطة العرض التوضيحي"""
    logger.info("\n--- 12. اختبار العرض التوضيحي ---")
    
    try:
        resp = post("/api/v1/analyze/demo", {})
        
        assert resp.get("success"), "العرض التوضيحي فشل"
        demo_results = resp.get("demo_results", [])
        
        assert len(demo_results) > 0, "لا توجد نتائج"
        
        for r in demo_results:
            assert "text" in r, "نص مفقود"
            assert "sentiment" in r, "مشاعر مفقودة"
            assert "confidence" in r, "ثقة مفقودة"
        
        results.success("العرض التوضيحي", f"{len(demo_results)} نتيجة")
        
    except Exception as e:
        results.fail("العرض التوضيحي", str(e))


def test_13_stats_endpoint():
    """اختبار إحصائيات الأداء"""
    logger.info("\n--- 13. اختبار الإحصائيات ---")
    
    try:
        resp = get("/api/v1/stats")
        
        assert "engine_stats" in resp, "إحصائيات المحرك مفقودة"
        assert "server_stats" in resp, "إحصائيات الخادم مفقودة"
        
        engine_stats = resp["engine_stats"]
        assert "model_loaded" in engine_stats
        assert "cache_stats" in engine_stats
        
        cache_stats = engine_stats["cache_stats"]
        assert "hit_rate" in cache_stats
        assert "size" in cache_stats
        
        results.success(
            "إحصائيات الأداء",
            f"requests={resp['server_stats']['total_requests']}, "
            f"cache_hit_rate={cache_stats['hit_rate']:.1%}"
        )
        
    except Exception as e:
        results.fail("الإحصائيات", str(e))


def test_14_error_handling():
    """اختبار معالجة الأخطاء"""
    logger.info("\n--- 14. اختبار معالجة الأخطاء ---")
    
    # Test empty text
    try:
        resp = requests.post(
            f"{BASE_URL}/api/v1/analyze",
            json={"text": "   "},
            timeout=10
        )
        # Should return error
        if resp.status_code == 422 or not resp.json().get("success"):
            results.success("رفض النص الفارغ", f"HTTP {resp.status_code}")
        else:
            results.warn("رفض النص الفارغ", "لم يُرفض النص الفارغ")
    except Exception as e:
        results.warn("رفض النص الفارغ", str(e))
    
    # Test 404
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/nonexistent", timeout=10)
        if resp.status_code == 404:
            results.success("معالجة 404", "حالة صحيحة")
        else:
            results.warn("معالجة 404", f"HTTP {resp.status_code}")
    except Exception as e:
        results.warn("معالجة 404", str(e))


# ============================
# Main Runner
# ============================

def run_all_tests():
    logger.info("=" * 60)
    logger.info("🧪 Sentiment Arabia - اختبارات شاملة")
    logger.info("=" * 60)
    
    # Check server is running
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/health", timeout=5)
        logger.info(f"✅ الخادم يعمل على {BASE_URL}")
    except Exception as e:
        logger.error(f"❌ الخادم غير متاح على {BASE_URL}")
        logger.error(f"   تأكد من تشغيل: python3 api/main.py")
        logger.error(f"   الخطأ: {e}")
        sys.exit(1)
    
    # Run all tests
    test_01_health()
    test_02_basic_sentiment()
    test_03_confidence_scores()
    test_04_edge_cases()
    test_05_consistency()
    test_06_caching()
    test_07_batch_processing()
    test_08_performance()
    test_09_companies_api()
    test_10_product_search()
    test_11_arabic_specific()
    test_12_demo_endpoint()
    test_13_stats_endpoint()
    test_14_error_handling()
    
    # Print summary
    success = results.summary()
    
    # Save results to JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "passed": results.passed,
        "failed": results.failed,
        "warnings": len(results.warnings),
        "success_rate": results.passed / (results.passed + results.failed) * 100,
        "errors": results.errors,
        "warnings_list": results.warnings,
        "performance": {
            "latencies": [p['latency'] for p in results.performance_data]
        }
    }
    
    report_path = Path("/home/user/webapp/logs/test_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n📄 تقرير الاختبارات محفوظ في: {report_path}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
