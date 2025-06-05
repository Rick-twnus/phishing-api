from flask import Flask, request, jsonify
import joblib
import pandas as pd
import re
import math
from urllib.parse import urlparse

app = Flask(__name__)

# 載入模型與標準化器
model = joblib.load('phishing_model.pkl')
scaler = joblib.load('scaler.pkl')

feature_names = [
    'url_length',
    'num_dots',
    'num_slashes',
    'dangerous_char_ratio',
    'numerical_char_ratio',
    'dangerous_tld',
    'entropy',
    'is_ip',
    'domain_length',
    'full_domain_length',
    'subdomain_count',
    'suspicious_keyword_ratio',
    'repetitions',
    'redirections',
    'brand_spoof_score',
    'whitelisted'
]

dangerous_chars = ['@', '#', '%', '&', '=', '+', '$']
suspicious_keywords = ['login', 'secure', 'update', 'verify', 'account', 'bank', 'paypal']
dangerous_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq']

def calculate_entropy(s):
    prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(s)]
    return -sum([p * math.log2(p) for p in prob])

def extract_features_from_url(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
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
    repetitions = max((len(list(g)) for _, g in re.groupby(full_url)), default=1) / url_length
    redirections = full_url.count('//') - 1
    brand_spoof_score = 1 if any(kw in hostname.lower() for kw in ['paypol', 'faceb00k', 'g00gle', 'micros0ft']) else 0
    tld = '.' + hostname.split('.')[-1] if '.' in hostname else ''
    dangerous_tld = 1 if tld in dangerous_tlds else 0
    whitelisted = 0

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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        features = extract_features_from_url(url)
        df = pd.DataFrame(features, columns=feature_names)
        features_scaled = scaler.transform(df)
        prediction = model.predict(features_scaled)
        return jsonify({'result': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
