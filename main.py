import os
import json
from math import log, sqrt, exp

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from scipy.stats import norm
from openai import AzureOpenAI
import yfinance as yf
import numpy as np

app = FastAPI()

# -----------------------------
# Black-Scholes d1, d2
# -----------------------------
def d1(S, K, T, r, sigma):
    return (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * sqrt(T)

# -----------------------------
# Black-Scholes price only
# -----------------------------
def bs_price(S, K, T, r, sigma, option_type):
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)

    if option_type == "call":
        return S * norm.cdf(D1) - K * exp(-r * T) * norm.cdf(D2)
    else:
        return K * exp(-r * T) * norm.cdf(-D2) - S * norm.cdf(-D1)

# -----------------------------
# Vega（IV計算に必要）
# -----------------------------
def bs_vega(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(D1) * sqrt(T)

# -----------------------------
# IV計算（ニュートン法）
# -----------------------------
def implied_volatility(S, K, T, r, market_price, option_type,
                       initial_sigma=0.2, tol=1e-6, max_iter=100):
    sigma = initial_sigma

    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        vega = bs_vega(S, K, T, r, sigma)
        if vega == 0:
            break

        sigma = sigma - diff / vega
        if sigma <= 0:
            sigma = tol

    return None

# -----------------------------
# Greeks
# -----------------------------
def greeks(S, K, T, r, sigma, option_type):
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)

    if option_type == "call":
        delta = norm.cdf(D1)
        price = S * norm.cdf(D1) - K * exp(-r * T) * norm.cdf(D2)
    else:
        delta = -norm.cdf(-D1)
        price = K * exp(-r * T) * norm.cdf(-D2) - S * norm.cdf(-D1)

    gamma = norm.pdf(D1) / (S * sigma * sqrt(T))
    vega = S * norm.pdf(D1) * sqrt(T)
    theta = (
        - (S * norm.pdf(D1) * sigma) / (2 * sqrt(T))
        - r * K * exp(-r * T) * norm.cdf(D2 if option_type == "call" else -D2)
    )
    rho = K * T * exp(-r * T) * norm.cdf(D2 if option_type == "call" else -D2)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho
    }

# -----------------------------
# GPT: IV戦略生成
# -----------------------------
def gpt_iv_strategy(iv, S, K, T, option_type):
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    prompt = f"""
あなたはオプション戦略の専門家であり、同時に初心者向けの講師でもあります。

以下の IV（インプライドボラティリティ）とオプション条件を分析し、
必ず「数値を使った根拠」を含めて説明してください。

特に次の数値を必ず使って解説すること：
- IV（例: 0.21 → 21%）
- BS価格（price）
- delta / gamma / theta / vega / rho
- 株価 S と ストライク K の位置関係
- 満期 T（年換算）
- 最大利益・最大損失の具体例（可能な場合）

出力内容：
1. 最適な戦略（例：ベアコール / ブルプット / ストラドル / ストラングル など）
2. 専門家としての判断理由（数値を使って 2〜4行）
3. 初心者向けに、できるだけ噛み砕いた解説（数値を使って 4〜8行）
4. 初心者が注意すべきポイント（数値を使って 1〜3行）
5. 読みが外れた場合の「次の一手（Plan B）」を提案（数値を使って 3〜6行）

返答は必ず次の JSON 形式のみ：

{{
  "strategy": "戦略名",
  "expert_reason": "専門家としての理由を数値入りで2〜4行",
  "beginner_explanation": "初心者向けに数値入りで4〜8行でわかりやすく解説",
  "beginner_caution": "初心者が注意すべきポイントを数値入りで1〜3行で",
  "next_step": "読みが外れた場合の次の一手を数値入りで3〜6行で"
}}

【IVデータ】
IV: {iv}
株価 S: {S}
ストライク K: {K}
満期 T: {T}
オプションタイプ: {option_type}
"""
  
    try:
        res = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = res.choices[0].message.content.strip()

        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        if json_start == -1 or json_end == -1:
            return {"error": "no_json_found", "raw": raw}

        json_text = raw[json_start:json_end]
        json_text = json_text.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(json_text)
        except Exception as e:
            return {"error": "json_parse_error", "exception": str(e), "raw": raw, "json_text": json_text}

        keys = ["strategy", "expert_reason", "beginner_explanation", "beginner_caution", "next_step"]
        safe_data = {k: data.get(k, "") for k in keys}
        return safe_data

    except Exception as e:
        return {"error": "api_exception", "exception": str(e)}

# -----------------------------
# API: Greeks
# -----------------------------
@app.get("/api/greeks")
def api_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str):
    return greeks(S, K, T, r, sigma, option_type)

# -----------------------------
# API: BS Price
# -----------------------------
@app.get("/api/bs_price")
def api_bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str):
    return {"price": greeks(S, K, T, r, sigma, option_type)["price"]}

# -----------------------------
# API: Historical Volatility（安定版）
# -----------------------------
@app.get("/api/vol/historical")
def api_historical_vol(ticker: str = "^N225", days: int = 20):
    try:
        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(period=f"{days+1}d")

        if len(hist) < days + 1:
            return {"volatility": None}

        close = hist["Close"].values
        log_returns = np.log(close[1:] / close[:-1])
        vol = float(np.std(log_returns) * np.sqrt(252))

        return {"ticker": ticker, "days": days, "volatility": vol}

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# 日経225の現在値（共通化）
# -----------------------------
@app.get("/api/nk225_params")
def nk225_params():
    try:
        yf_ticker = yf.Ticker("^N225")
        info = yf_ticker.info

        return {
            "price": info.get("regularMarketPrice"),
            "previous_close": info.get("regularMarketPreviousClose")
        }
    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# API: IV計算
# -----------------------------
@app.get("/api/iv")
def api_iv(S: float,
           K: float,
           T: float,
           r: float,
           market_price: float,
           option_type: str,
           initial_sigma: float = 0.2):
    iv = implied_volatility(S, K, T, r, market_price, option_type,
                            initial_sigma=initial_sigma)

    if iv is None:
        return {"error": "IVが収束しませんでした"}

    return {"iv": iv}

# -----------------------------
# API: IV戦略（GPT）
# -----------------------------
@app.get("/api/iv_strategy")
def api_iv_strategy(iv: float, S: float, K: float, T: float, option_type: str):
    return gpt_iv_strategy(iv, S, K, T, option_type)

# -----------------------------
# UI : 自動計算版 + IV計算 + IV戦略
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>stock-learning-api</title>

<style>
  :root{
    --bg:#ffffff;
    --panel:#f2f2f2;
    --accent:#0078ff;
    --text:#000;
  }
  body{
    margin:0;
    background:var(--bg);
    color:var(--text);
    font-family:system-ui, -apple-system, "Hiragino Kaku Gothic ProN", sans-serif;
    padding:16px;
    font-size:22px;
  }
  h2, h3{
    font-size:28px;
    margin-bottom:12px;
  }
  select, input{
    width:100%;
    font-size:24px;
    padding:16px;
    margin:10px 0;
    border-radius:10px;
    border:1px solid #ccc;
    background:#fff;
  }
  button{
    width:100%;
    font-size:26px;
    padding:18px;
    border-radius:12px;
    margin-top:16px;
    background:var(--accent);
    color:#fff;
    border:none;
  }
  #resultBox, #ivBox, #ivStrategyBox{
    background:var(--panel);
    padding:16px;
    border-radius:10px;
    font-size:24px;
    margin-top:16px;
  }
</style>
</head>

<body>

<h2>stock-learning-api</h2>

<h3>入力</h3>

株価 S:<br>
<input id="S" type="number">

ストライク K:<br>
<input id="K" type="number" value="70000">

満期 T（年換算）:<br>
<input id="T" type="number" value="0.1">

金利 r:<br>
<input id="r" type="number" value="0.001">

ボラティリティ σ:<br>
<input id="sigma" type="number" value="0.20">

オプションタイプ:<br>
<select id="option_type">
  <option value="call">コール</option>
  <option value="put">プット</option>
</select>

<button onclick="loadSummary()">計算する</button>

<div id="resultBox"></div>

<hr>

<h3>IV計算</h3>

市場価格（オプション価格）:<br>
<input id="market_price" type="number" placeholder="例: 1800">

<button onclick="loadIV()">IVを計算する</button>

<div id="ivBox"></div>

<hr>

<h3>IV戦略（GPT）</h3>

<button onclick="loadIVStrategy()">IV戦略を表示する</button>

<div id="ivStrategyBox"></div>

<script>
async function loadNK225(){
    try{
        const res = await fetch("/api/nk225_params");
        const data = await res.json();
        if(data.price){
            document.getElementById("S").value = data.price;
        }
    }catch(e){
        console.log("NK225取得エラー:", e);
    }
}

async function loadSummary(){
    const S = document.getElementById("S").value;
    const K = document.getElementById("K").value;
    const T = document.getElementById("T").value;
    const r = document.getElementById("r").value;
    const sigma = document.getElementById("sigma").value;
    const option_type = document.getElementById("option_type").value;

    const greeksUrl = `/api/greeks?S=${S}&K=${K}&T=${T}&r=${r}&sigma=${sigma}&option_type=${option_type}`;
    const priceUrl  = `/api/bs_price?S=${S}&K=${K}&T=${T}&r=${r}&sigma=${sigma}&option_type=${option_type}`;
    const hvUrl     = `/api/vol/historical?days=20`;

    const [greeks, price, hv] = await Promise.all([
        fetch(greeksUrl).then(r => r.json()),
        fetch(priceUrl).then(r => r.json()),
        fetch(hvUrl).then(r => r.json())
    ]);

    document.getElementById("resultBox").innerHTML = `
📌 株価 S: ${S}<br>
📌 ボラティリティ σ: ${sigma}<br><br>

<b>【Greeks】</b><br>
delta: ${greeks.delta}<br>
gamma: ${greeks.gamma}<br>
theta: ${greeks.theta}<br>
vega: ${greeks.vega}<br>
rho: ${greeks.rho}<br><br>

<b>【BS価格】</b><br>
price: ${price.price}<br><br>

<b>【ヒストリカルボラ（20日）】</b><br>
volatility: ${hv.volatility ?? "データなし"}
    `;
}

async function loadIV(){
    const S = document.getElementById("S").value;
    const K = document.getElementById("K").value;
    const T = document.getElementById("T").value;
    const r = document.getElementById("r").value;
    const market_price = document.getElementById("market_price").value;
    const option_type = document.getElementById("option_type").value;

    const url = `/api/iv?S=${S}&K=${K}&T=${T}&r=${r}&market_price=${market_price}&option_type=${option_type}`;

    const iv = await fetch(url).then(r => r.json());

    document.getElementById("ivBox").innerHTML = `
<b>【IV（インプライド・ボラティリティ）】</b><br>
${iv.iv ? iv.iv : (iv.error ? iv.error : "計算できませんでした")}
    `;
}

async function loadIVStrategy(){
    const S = document.getElementById("S").value;
    const K = document.getElementById("K").value;
    const T = document.getElementById("T").value;
    const option_type = document.getElementById("option_type").value;

    const ivText = document.getElementById("ivBox").innerText;
    const ivMatch = ivText.match(/([0-9.]+)/);
    if(!ivMatch){
        document.getElementById("ivStrategyBox").innerHTML = "先に IV を計算してください。";
        return;
    }
    const iv = parseFloat(ivMatch[1]);

    const url = `/api/iv_strategy?iv=${iv}&S=${S}&K=${K}&T=${T}&option_type=${option_type}`;
    const strategy = await fetch(url).then(r => r.json());

    if(strategy.error){
        document.getElementById("ivStrategyBox").innerHTML = "戦略生成でエラーが発生しました。";
        return;
    }

    document.getElementById("ivStrategyBox").innerHTML = `
<b>【IV戦略】</b><br>
戦略: ${strategy.strategy}<br><br>

<b>専門家の判断理由</b><br>
${strategy.expert_reason}<br><br>

<b>初心者向けの解説</b><br>
${strategy.beginner_explanation}<br><br>

<b>注意ポイント</b><br>
${strategy.beginner_caution}<br><br>

<b>次の一手（Plan B）</b><br>
${strategy.next_step}
    `;
}

window.onload = async () => {
    await loadNK225();
    await loadSummary();
};
</script>

</body>
</html>
"""
