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
# AI戦略名をレッグ構成に変換する
# -----------------------------
def strategy_to_legs(strategy_name, S, K, iv):
    """
    AI戦略名をレッグ構成に変換する
    """
    if strategy_name == "bullish call spread":
        return [
            { "option_type": "call", "position": "long",  "K": K,       "quantity": 1, "premium": 0 },
            { "option_type": "call", "position": "short", "K": K + 1000, "quantity": 1, "premium": 0 }
        ]

    if strategy_name == "bear put spread":
        return [
            { "option_type": "put", "position": "long",  "K": K,       "quantity": 1, "premium": 0 },
            { "option_type": "put", "position": "short", "K": K - 1000, "quantity": 1, "premium": 0 }
        ]

    if strategy_name == "long call":
        return [
            { "option_type": "call", "position": "long", "K": K, "quantity": 1, "premium": 0 }
        ]

    if strategy_name == "long put":
        return [
            { "option_type": "put", "position": "long", "K": K, "quantity": 1, "premium": 0 }
        ]

    return None

# -----------------------------
# GPT: IV戦略生成
# -----------------------------
def gpt_iv_strategy(iv, S, K, T, option_type,
                   delta, gamma, theta, vega, rho,
                   bs_price_value,
                   scenario_text, time_scenario_text):

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    # None対策
    scenario_text = str(scenario_text)
    time_scenario_text = str(time_scenario_text)

    # f-string の {} をすべて {{ }} にエスケープ
    prompt = f"""
あなたはプロのオプション戦略アナリストです。

以下のデータを使って戦略を判断してください。
必ず「数値に基づく理由」を書いてください。

【市場データ】
株価 S: {S}
ストライク K: {K}
満期 T: {T}
オプションタイプ: {option_type}

【IV】
IV: {iv}

【Greeks】
delta: {delta}
gamma: {gamma}
theta: {theta}
vega: {vega}
rho: {rho}

【BS理論価格】
price: {bs_price_value}

【株価シナリオ（±5%、±10%）】
{scenario_text}

【時間軸シナリオ（±5%、±10%）】
{time_scenario_text}

【重要】
- JSON 以外の文章を一切書かない
- コードブロック（```）禁止
- JSON の前後に説明文を絶対に書かない
- JSON が壊れていたら再生成する

【出力形式】
次の JSON のみを返す：

{{
  "strategy": "",
  "expert_reason": "",
  "beginner_explanation": "",
  "beginner_caution": "",
  "next_step": ""
}}
"""

    try:
        res = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        raw = res.choices[0].message.content.strip()

        # JSON抽出ロジック強化
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1

        if json_start == -1 or json_end == -1:
            return {
                "strategy": "error",
                "expert_reason": "JSONが見つかりませんでした",
                "beginner_explanation": raw,
                "beginner_caution": "",
                "next_step": ""
            }

        json_text = raw[json_start:json_end]
        json_text = json_text.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(json_text)
        except Exception as e:
            return {
                "strategy": "error",
                "expert_reason": "JSON解析エラー",
                "beginner_explanation": str(e),
                "beginner_caution": json_text,
                "next_step": ""
            }

        keys = ["strategy", "expert_reason", "beginner_explanation", "beginner_caution", "next_step"]
        safe_data = {k: data.get(k, "") for k in keys}

        return safe_data

    except Exception as e:
        return {
            "strategy": "error",
            "expert_reason": "API例外",
            "beginner_explanation": str(e),
            "beginner_caution": "",
            "next_step": ""
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
def bs_price_api(S: float, K: float, T: float, r: float, sigma: float, option_type: str):
    price = bs_price(S, K, T, r, sigma, option_type)
    return {"price": price}

# -----------------------------
# API: シナリオBS価格（±5%、±10%）
# -----------------------------
@app.get("/api/scenario_bs")
def scenario_bs_api(S: float, K: float, T: float, r: float, sigma: float, option_type: str):
    scenarios = [
        ("現在値", 0.00),
        ("+5%",   0.05),
        ("-5%",  -0.05),
        ("+10%",  0.10),
        ("-10%", -0.10),
    ]

    results = []
    for label, rate in scenarios:
        S_scenario = S * (1 + rate)
        price = bs_price(S_scenario, K, T, r, sigma, option_type)
        results.append({
            "label": label,
            "S": S_scenario,
            "price": price
        })

    return {"scenarios": results}

# -----------------------------
# API: Historical Volatility
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
# 日経225の現在値
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

    # まず Greeks を計算
    g = greeks(S, K, T, 0.001, 0.3, option_type)

    delta = g["delta"]
    gamma = g["gamma"]
    theta = g["theta"]
    vega  = g["vega"]
    rho   = g["rho"]

    # BS理論価格
    bs_price_value = g["price"]

    # シナリオ（±5%、±10%）
    scenario_text = "..."  # ← ここは後で自動生成

    # 時間軸シナリオ（±5%、±10%）
    time_scenario_text = "..."  # ← ここも後で自動生成

    return gpt_iv_strategy(
        iv, S, K, T, option_type,
        delta, gamma, theta, vega, rho,
        bs_price_value,
        scenario_text,
        time_scenario_text
    )

# -----------------------------
# API: 時間軸シナリオ（±5%、±10%）
# -----------------------------
@app.get("/api/time_scenario_bs")
def time_scenario_bs(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    days: int = 7
):
    # 残存期間を更新（年換算）
    T_new = T - days / 365
    if T_new < 0:
        T_new = 0.00001

    # 株価変動シナリオ（±5%、±10%）
    scenarios = [
        ("+5%",   0.05),
        ("-5%",  -0.05),
        ("+10%",  0.10),
        ("-10%", -0.10),
    ]

    results = []

    for label, rate in scenarios:
        S_new = S * (1 + rate)
        price = bs_price(S_new, K, T_new, r, sigma, option_type)

        results.append({
            "label": f"{days}日後 {label}",
            "S": S_new,
            "T_new": T_new,
            "price": price
        })

    return {"scenarios": results}


# -----------------------------
# 戦略シミュレーションAPI（手入力レッグ専用）
# -----------------------------
from pydantic import BaseModel
from typing import List

class Leg(BaseModel):
    option_type: str   # call / put
    position: str      # long / short
    K: float           # strike
    quantity: int = 1  # 枚数
    premium: float = 0.0  # プレミアム（買いは支払い、売りは受け取り）

class StrategyRequest(BaseModel):
    S: float        # 現在の株価
    T: float        # 満期（年換算）
    r: float        # 金利
    sigma: float    # IV
    legs: List[Leg] # 手入力レッグ一覧

@app.post("/api/strategy_simulation")
def api_strategy_simulation(req: StrategyRequest):

    pl_curve = []
    max_profit = -999999
    max_loss = 999999

    # -----------------------------
    # 初期コスト（プレミアム合計）
    # -----------------------------
    initial_cost = 0.0
    for leg in req.legs:
        if leg.position == "long":
            initial_cost -= leg.premium * leg.quantity
        else:
            initial_cost += leg.premium * leg.quantity

    # -----------------------------
    # 満期時の株価レンジ（±20%）
    # -----------------------------
    S_min = req.S * 0.8
    S_max = req.S * 1.2
    steps = 50
    dS = (S_max - S_min) / steps

    for i in range(steps + 1):
        S_T = S_min + dS * i
        total_pl = 0.0

        # -----------------------------
        # 各レッグの満期価値（BS理論価格）
        # -----------------------------
        for leg in req.legs:
            price = bs_price(S_T, leg.K, req.T, req.r, req.sigma, leg.option_type)

            if leg.position == "long":
                total_pl += price * leg.quantity
            else:
                total_pl -= price * leg.quantity

        # 初期コストを反映
        total_pl += initial_cost

        pl_curve.append({"S_T": S_T, "profit": total_pl})

        max_profit = max(max_profit, total_pl)
        max_loss = min(max_loss, total_pl)

    # -----------------------------
    # 損益分岐点（初めて利益が0以上になる株価）
    # -----------------------------
    breakeven = None
    for p in pl_curve:
        if p["profit"] >= 0:
            breakeven = p["S_T"]
            break

    return {
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "pl_curve": pl_curve
    }

# -----------------------------
# UI : シンプル版（スプレッドセット削除）
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
  #resultBox, #ivBox, #ivStrategyBox, #scenarioBox, #simBox{
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

満期（選択）:<br>
<select id="T_weeks" onchange="updateT()">
  <option value="1">1週間</option>
  <option value="2">2週間</option>
  <option value="3">3週間</option>
  <option value="4">4週間</option>
  <option value="5">5週間</option>
  <option value="6">6週間</option>
</select>

<!-- 年換算T（内部用） -->
<input id="T" type="hidden" value="0.0192">

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

<h3>株価シナリオ（±5%、±10%）</h3>
<button onclick="loadScenarioBS()">シナリオ計算を実行する</button>
<div id="scenarioBox"></div>

<hr>

<h3>時間軸シナリオ（例：1週間後）</h3>

日数:<br>
<input id="days" type="number" value="7">

<button onclick="loadTimeScenario()">時間軸シナリオを計算する</button>

<div id="timeScenarioBox"></div>

<hr>

<h3>IV計算</h3>

市場価格（オプション価格）:<br>
<input id="market_price" type="number" placeholder="例: 1800">

<button onclick="loadIV()">IVを計算する</button>

<div id="ivBox"></div>

<hr>

<h3>IV戦略（AI）</h3>

<button onclick="loadIVStrategy()">IV戦略(AI)を表示する</button>

<div id="ivStrategyBox"></div>

<!-- ★ 追加：AI戦略を戦略シミュレーションへ連動 -->
<button onclick="runStrategySimulation()">AI戦略をシミュレーションする</button>
<div id="aiSimBox"></div>

<hr>

<h3>戦略シミュレーション（手入力レッグ専用）</h3>

<b>レッグ入力（JSON形式）</b><br>
<textarea id="legsInput" style="width:100%; height:200px; font-size:20px;">
[
  { "option_type": "call", "position": "long",  "K": 70000, "quantity": 1, "premium": 1800 },
  { "option_type": "call", "position": "short", "K": 71000, "quantity": 1, "premium": 1200 }
]
</textarea>

<button onclick="runXXSimulation()">戦略シミュレーションを実行する</button>

<div id="simBox"></div>

</body>


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

<b>【オプション理論価格（BSモデル）】</b><br>
price: ${price.price}<br><br>

<b>【ヒストリカルボラ（20日）】</b><br>
volatility: ${hv.volatility ?? "データなし"}
    `;
}

function updateT(){
    const weeks = parseInt(document.getElementById("T_weeks").value);
    const T_year = (weeks * 7) / 365;
    document.getElementById("T").value = T_year.toFixed(4);
}

async function loadTimeScenario(){
    const S = document.getElementById("S").value;
    const K = document.getElementById("K").value;
    const T = document.getElementById("T").value;
    const r = document.getElementById("r").value;
    const sigma = document.getElementById("sigma").value;
    const option_type = document.getElementById("option_type").value;
    const days = document.getElementById("days").value;

    const url = `/api/time_scenario_bs?S=${S}&K=${K}&T=${T}&r=${r}&sigma=${sigma}&option_type=${option_type}&days=${days}`;
    const data = await fetch(url).then(r => r.json());

    let html = "<table border='1' style='width:100%; font-size:22px;'>";
    html += "<tr><th>シナリオ</th><th>株価</th><th>残存T</th><th>理論価格</th></tr>";

    for(const row of data.scenarios){
        html += `
            <tr>
                <td>${row.label}</td>
                <td>${row.S.toFixed(2)}</td>
                <td>${row.T_new.toFixed(4)}</td>
                <td>${row.price.toFixed(2)}</td>
            </tr>
        `;
    }

    html += "</table>";

    document.getElementById("timeScenarioBox").innerHTML = html;
}

async function loadScenarioBS(){
    const S = document.getElementById("S").value;
    const K = document.getElementById("K").value;
    const T = document.getElementById("T").value;
    const r = document.getElementById("r").value;
    const sigma = document.getElementById("sigma").value;
    const option_type = document.getElementById("option_type").value;

    const url = `/api/scenario_bs?S=${S}&K=${K}&T=${T}&r=${r}&sigma=${sigma}&option_type=${option_type}`;
    const data = await fetch(url).then(r => r.json());

    let html = "<table border='1' style='width:100%; font-size:22px;'>";
    html += "<tr><th>シナリオ</th><th>株価</th><th>理論価格</th></tr>";

    for(const row of data.scenarios){
        html += `
            <tr>
                <td>${row.label}</td>
                <td>${row.S.toFixed(2)}</td>
                <td>${row.price.toFixed(2)}</td>
            </tr>
        `;
    }

    html += "</table>";

    document.getElementById("scenarioBox").innerHTML = html;
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

    // ★ AI戦略を保存（これが無いと runStrategySimulation が動かない）
    window.lastStrategy = strategy;

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

async function runStrategySimulation(){
    const strategy = window.lastStrategy;  // AI戦略のJSON
    if(!strategy || !strategy.legs){
        alert("AI戦略がまだ生成されていません");
        return;
    }

    const S = parseFloat(document.getElementById("S").value);
    const T = parseFloat(document.getElementById("T").value);
    const r = parseFloat(document.getElementById("r").value);
    const sigma = parseFloat(document.getElementById("sigma").value);

    const body = {
        S, T, r, sigma,
        legs: strategy.legs
    };

    const res = await fetch("/api/strategy_simulation", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
    });

    const data = await res.json();

    document.getElementById("simBox").innerHTML = `
<b>【AI戦略シミュレーション】</b><br>
戦略: ${strategy.strategy}<br><br>

最大利益: ${data.max_profit}<br>
最大損失: ${data.max_loss}<br>
損益分岐点: ${data.breakeven}<br><br>

<b>損益曲線（最初の10点）</b><br>
${data.pl_curve.slice(0,10).map(p => `S=${Math.round(p.S_T)} → 利益=${p.profit}`).join("<br>")}
    `;
}

async function runXXSimulation(){
    let legsJson = document.getElementById("legsInput").value;

    try{
        window.currentLegs = JSON.parse(legsJson);
    }catch(e){
        document.getElementById("simBox").innerHTML = "レッグ入力がJSONとして読み込めません。";
        return;
    }

    const S = parseFloat(document.getElementById("S").value);
    const T = parseFloat(document.getElementById("T").value);
    const r = parseFloat(document.getElementById("r").value);
    const sigma = parseFloat(document.getElementById("sigma").value);

    const body = { S, T, r, sigma, legs: window.currentLegs };

    const res = await fetch("/api/strategy_simulation", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
    });

    const data = await res.json();

    document.getElementById("simBox").innerHTML = `
<b>【戦略シミュレーション】</b><br>
最大利益: ${data.max_profit}<br>
最大損失: ${data.max_loss}<br>
損益分岐点: ${data.breakeven}<br><br>

<b>損益曲線（最初の10点）</b><br>
${data.pl_curve.slice(0,10).map(p => `S=${Math.round(p.S_T)} → 利益=${p.profit}`).join("<br>")}
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
