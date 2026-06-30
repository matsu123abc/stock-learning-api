from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from math import log, sqrt, exp
from scipy.stats import norm
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
            return {"error": "データ不足"}

        close = hist["Close"].values
        log_returns = np.log(close[1:] / close[:-1])
        vol = float(np.std(log_returns) * np.sqrt(252))

        return {"ticker": ticker, "days": days, "volatility": vol}

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# ② 日経225の現在値（共通化）
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
# UI : 自動計算版 + IV計算UI追加
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
  #resultBox, #ivBox{
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

市場価格（market price）:<br>
<input id="market_price" type="number" placeholder="例: 1800">

<button onclick="loadIV()">IVを計算する</button>

<div id="ivBox"></div>

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

    const greeks = await fetch(greeksUrl).then(r=>r.json());
    const price  = await fetch(priceUrl).then(r=>r.json());
    const hv     = await fetch(hvUrl).then(r=>r.json());

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
volatility: ${hv.volatility}
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
${iv.iv ? iv.iv : "計算できませんでした"}
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
