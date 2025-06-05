from flask import Flask, request, jsonify
import pandas as pd
import re
import joblib
import os
from collections import Counter
from scipy.stats import entropy
import tldextract
from Levenshtein import distance as levenshtein_distance

# ===== 載入模型與 scaler =====
model = joblib.load("phishing_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===== 特徵定義 =====
dangerous_chars = ['@', '?', '-', '=', '&', '%']
dangerous_TLDs = ['tk', 'ml', 'ga', 'cf', 'gq']
sus_words = ['secure', 'account', 'update', 'login', 'verify', 'signin', 'bank', 'notify', 'click', 'inconvenient']
ip_pattern = r'[0-9]+(?:\.[0-9]+){3}'
whitelist = ['google', 'youtube', 'facebook', 'twitter', 'wikipedia', 'microsoft', 'amazon', 'apple']
common_brands = ["google", "facebook", "paypal", "amazon", "apple", "microsoft", "youtube", "netflix", "twitter", "instagram", "linkedin", "github", "dropbox"]
confusables = {'0': 'o', '1': 'l', '3': 'e', '5': 's', '7': 't', '8': 'b', '9': 'g', 'l': 'i', 'rn': 'm'}

def normalize_confusables(s):
    for k, v in confusables.items():
        s = s.replace(k, v)
    return s

def brand_spoof_score(domain):
    norm_domain = normalize_confusables(domain.lower())
    return min(levenshtein_distance(norm_domain, brand) for brand in common_brands)

def urlentropy(url):
    if not url or len(url) == 0:
        return 0.0
    frequencies = Counter(url)
    prob = [count / len(url) for count in frequencies.values()]
    return entropy(prob, base=2)

def redirection(url):
    pos = url.rfind('//')
    return 1.0 if pos > 7 else 0.0

# ===== 特徵擷取函數 =====
def extract_features(df):
    X = pd.DataFrame()
    tld_data = df['URL'].apply(tldextract.extract)

    X['URL length'] = df['URL'].apply(len)
    X['Number of dots'] = df['URL'].apply(lambda x: x.count('.'))
    X['Number of slashes'] = df['URL'].apply(lambda x: x.count('/'))
    X['Dangerous char ratio'] = df['URL'].apply(lambda x: sum(c in dangerous_chars for c in x) / len(x))
    X['Numerical char ratio'] = df['URL'].apply(lambda x: sum(c.isdigit() for c in x) / len(x))
    X['Dangerous TLD'] = tld_data.apply(lambda x: x.suffix in dangerous_TLDs).astype(float)
    X['Entropy'] = df['URL'].apply(urlentropy)
    X['IP Address'] = df['URL'].apply(lambda x: bool(re.search(ip_pattern, x))).astype(float)
    X['Domain name length'] = tld_data.apply(lambda x: len(x.domain))
    X['Full domain length'] = tld_data.apply(lambda x: len(x.domain + '.' + x.suffix))
    X['Subdomain count'] = tld_data.apply(lambda x: len(x.subdomain.split('.')) if x.subdomain else 0)
    X['Suspicious keyword ratio'] = df['URL'].apply(lambda x: sum(word in x.lower() for word in sus_words) / len(x))
    X['Repetitions'] = tld_data.apply(lambda x: bool(re.search(r'(.)\1{2,}', x.domain))).astype(float)
    X['Redirections'] = df['URL'].apply(redirection)
    X['Brand Spoof Score'] = tld_data.apply(lambda x: brand_spoof_score(x.domain))
    X['Whitelisted'] = tld_data.apply(lambda x: x.registered_domain.lower() in whitelist).astype(float)

    return X.astype(float)

# ===== 建立 Flask 應用 =====
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    url = data.get("url")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        df = pd.DataFrame({"URL": [url]})
        features = extract_features(df)
        features_scaled = scaler.transform(features)
        prediction = int(model.predict(features_scaled)[0])
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # for Render deployment
    app.run(host="0.0.0.0", port=port)


