import os, json, datetime, math
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

try:
    import google.generativeai as genai
except Exception:
    genai = None

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def get_basic(ticker):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="6mo", interval="1d", auto_adjust=True)
    info = {}
    news = []
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    try:
        news = tk.news[:5]
    except Exception:
        news = []
    return hist, info, news

def analyze_ticker(ticker):
    hist, info, news = get_basic(ticker)
    if hist.empty:
        return {"ticker": ticker, "error": "No price data"}
    close = hist["Close"]
    price = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else price
    chg = (price / prev - 1) * 100 if prev else 0
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float("nan")
    rsi14 = float(rsi(close).iloc[-1]) if len(close) >= 20 else float("nan")
    high_52 = info.get("fiftyTwoWeekHigh")
    low_52 = info.get("fiftyTwoWeekLow")
    market_cap = info.get("marketCap")
    pe = info.get("trailingPE") or info.get("forwardPE")
    beta = info.get("beta")
    trend_score = 50
    trend_score += 10 if price > ma20 else -10
    if not math.isnan(ma50):
        trend_score += 10 if price > ma50 else -10
    if not math.isnan(rsi14):
        if 45 <= rsi14 <= 65: trend_score += 5
        elif rsi14 > 75: trend_score -= 10
        elif rsi14 < 35: trend_score += 5
    if abs(chg) > 5: trend_score -= 5
    trend_score = max(0, min(100, trend_score))
    headlines = []
    for n in news:
        title = n.get("title") or ""
        publisher = n.get("publisher") or ""
        link = n.get("link") or ""
        if title:
            headlines.append({"title": title, "publisher": publisher, "link": link})
    return {
        "ticker": ticker, "price": round(price, 2), "day_change_pct": round(chg, 2),
        "ma20": round(ma20, 2), "ma50": None if math.isnan(ma50) else round(ma50, 2),
        "rsi14": None if math.isnan(rsi14) else round(rsi14, 1),
        "trend_score": trend_score, "market_cap": market_cap, "pe": pe, "beta": beta,
        "fiftyTwoWeekHigh": high_52, "fiftyTwoWeekLow": low_52, "headlines": headlines
    }

def ai_summary(results):
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key or genai is None:
        return "未启用 Gemini：请在 GitHub Secrets 添加 GEMINI_API_KEY。"
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
你是用户的加拿大/美股 AI 投资研究助手。请基于以下结构化数据生成中文日报。
要求：
1. 不承诺收益，不给绝对买卖指令。
2. 按“核心持仓、观察名单、风险提醒、今日最值得关注”输出。
3. 特别关注 AI光通信、航天、半导体、AI制药、稳定币。
4. 给每只股票一句话结论，并给出观察/持有/谨慎/等待回调等非强制动作。
数据：
{json.dumps(results, ensure_ascii=False, indent=2)}
"""
    try:
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"Gemini 分析失败：{e}"

def main():
    with open(ROOT / "tickers.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    all_tickers = cfg.get("portfolio", []) + cfg.get("watchlist", [])
    results = [analyze_ticker(t) for t in all_tickers]
    summary = ai_summary(results)
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    md = [f"# AI 投资日报 {today}\n", "## AI 总结\n", summary, "\n## 股票数据\n"]
    for r in results:
        if "error" in r:
            md.append(f"### {r['ticker']}\n- 错误：{r['error']}\n")
            continue
        md.append(f"### {r['ticker']}\n")
        md.append(f"- 价格：{r['price']}，日涨跌：{r['day_change_pct']}%\n")
        md.append(f"- 趋势评分：{r['trend_score']}/100，RSI14：{r['rsi14']}，MA20：{r['ma20']}，MA50：{r['ma50']}\n")
        md.append(f"- PE：{r['pe']}，Beta：{r['beta']}，52周：{r['fiftyTwoWeekLow']} - {r['fiftyTwoWeekHigh']}\n")
        if r["headlines"]:
            md.append("- 新闻：\n")
            for h in r["headlines"][:3]:
                md.append(f"  - {h['title']} ({h['publisher']})\n")
        md.append("\n")
    out = REPORT_DIR / f"daily_report_{today}.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(out)
    print("Report generated.")

if __name__ == "__main__":
    main()
