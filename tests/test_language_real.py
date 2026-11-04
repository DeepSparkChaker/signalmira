"""Actual Lingua language tests, skipped unless the optional dependency is installed."""

import importlib.util
import pytest

from enrichment.language import detect_text

pytestmark = pytest.mark.skipif(importlib.util.find_spec('lingua') is None,
                                 reason='Install: uv add lingua-language-detector')

@pytest.mark.parametrize('language,text', [
 ('en','My payment failed several times this week and customer support did not help.'),
 ('es','El pago falla constantemente, la aplicación se cierra y nadie responde al teléfono.'),
 ('fr','Le paiement échoue constamment et le service client ne répond pas à mes demandes.'),
 ('ar','عملية الدفع تفشل باستمرار منذ الأسبوع الماضي وخدمة العملاء لم تساعدني في حل المشكلة.'),
 ('zh','支付一直失败，手机应用程序经常崩溃，客户服务人员也无法帮助我解决问题。'),
])
def test_full_sentence_multilingual_detection(language,text):
    result=detect_text(text)
    assert result.language==language, result
