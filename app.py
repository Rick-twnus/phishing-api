from flask import Flask, request, jsonify
import joblib
import pandas as pd
import re
import tldextract
from collections import Counter
from scipy.stats import entropy

app = Flask(__name__)

# ===== 載入模型與標準化器 =====
model = joblib.load("phishing_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===== 特徵擷取函數 =====
dangerous_chars = ['@', '?', '-', '=', '&', '%']
dangerous_TLDs = ['tk', 'ml', 'ga', 'cf', 'gq']
sus_words = ['secure', 'account', 'update', 'login', 'verify', 'signin', 'bank', 'notify', 'click', 'inconvenient']
ip_pattern = r'[0-9]+(?:\.[0-9]+){3}'
whitelist = ['google', 'youtube', 'facebook', 'twitter', 'wikipedia', 'microsoft', 'amazon', 'apple']

def urlentropy(url):
    if not url:
        return 0.0
    frequencies = Counter(url)
    prob = [count / len(url) for count in frequencies.values()]
    return entropy(prob, base=2)

def redirection(url):
    pos = url.rfind('//')
    return 1.0 if pos > 7 else 0.0

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
    X['Brand Spoof Score'] = tld_data.apply(lambda x: len(x.domain))  # 可以放真實函數
    X['Whitelisted'] = tld_data.apply(lambda x: x.registered_domain.lower() in whitelist).astype(float)

    return X.astype(float)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    url = data.get("url")
    
    if not url or not isinstance(url, str) or not url.startswith("http"):
        return jsonify({"error": "Invalid URL"}), 400

    df = pd.DataFrame({"URL": [url]})
    features = extract_features(df)
    features_scaled = scaler.transform(features)

    prediction = int(model.predict(features_scaled)[0])
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)



