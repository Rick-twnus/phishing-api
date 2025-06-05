import re
import math
from urllib.parse import urlparse
import itertools

dangerous_chars = ['@', '#', '%', '&', '=', '+', '$']
suspicious_keywords = ['login', 'secure', 'update', 'verify', 'account', 'bank', 'paypal']
dangerous_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq']

def calculate_entropy(s):
    prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(s)]
    return -sum([p * math.log2(p) for p in prob])

def extract_features_from_url(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    path = parsed.path or ''
    full_url = url

    url_length = len(full_url)
    num_dots = full_url.count('.')
    num_slashes = full_url.count('/')
    dangerous_char_ratio = sum(c in dangerous_chars for c in full_url) / url_length
    numerical_char_ratio = sum(c.isdigit() for c in full_url) / url_length
    entropy = calculate_entropy(full_url)
    is_ip = bool(re.fullmatch(r'\d+\.\d+\.\d+\.\d+', hostname))
    domain_parts = hostname.split('.') if hostname else []
    subdomain_count = max(len(domain_parts) - 2, 0)
    domain_length = len(domain_parts[-2]) if len(domain_parts) >= 2 else 0
    full_domain_length = len(hostname)
    suspicious_keyword_ratio = sum(kw in full_url.lower() for kw in suspicious_keywords) / len(suspicious_keywords)
    repetitions = max((len(list(g)) for _, g in itertools.groupby(full_url)), default=1) / url_length
    redirections = full_url.count('//') - 1
    brand_spoof_score = 1 if any(kw in hostname.lower() for kw in ['paypol', 'faceb00k', 'g00gle', 'micros0ft']) else 0
    tld = '.' + hostname.split('.')[-1] if '.' in hostname else ''
    dangerous_tld = 1 if tld in dangerous_tlds else 0
    whitelisted = 0  # 實際部署時可改為查資料庫

    return [[
        url_length,
        num_dots,
        num_slashes,
        dangerous_char_ratio,
        numerical_char_ratio,
        dangerous_tld,
        entropy,
        is_ip,
        domain_length,
        full_domain_length,
        subdomain_count,
        suspicious_keyword_ratio,
        repetitions,
        redirections,
        brand_spoof_score,
        whitelisted
    ]]
