# 💼 Stock Portfolio App

> **A personal Indian stock portfolio tracker built with Streamlit — featuring real-time prices, AI-powered insights, sector analytics, and tax estimates, all in one elegant dashboard.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📌 Overview

**Stock Portfolio App** is a full-featured, interactive portfolio management tool tailored for Indian equity investors. Built entirely with Python and Streamlit, it lets you import holdings from popular brokers like **Groww**, **Zerodha**, and **Upstox**, fetch live NSE/BSE prices via Yahoo Finance, visualize your portfolio with beautiful charts, and even chat with an AI assistant about your investments.

No cloud database needed — everything runs locally with a lightweight SQLite backend.

---

## ✨ Features

### 📊 Dashboard
- Live metrics: **Total Investment**, **Current Value**, **Overall P&L**, and **Number of Holdings**
- **Sector-wise Donut Chart** — see how your capital is distributed across sectors
- **Top 10 Holdings Bar Chart** — color-coded by P&L percentage
- **Portfolio Heatmap (Treemap)** — visualize every holding by value and performance at a glance
- One-click **Refresh Data** button with 5-minute price caching

### 🗃️ Portfolio Manager
- **Import broker exports** — supports CSV (Groww, Upstox) and Excel (Zerodha) formats
- **Auto-detection of column names** — handles variations like `Qty`, `Quantity`, `Average buy price`, etc.
- **Smart ticker normalization** — maps Indian company names to Yahoo Finance NSE tickers automatically
- **Manual edit mode** — add, edit, or remove holdings directly in an interactive data table
- Persistent storage via **SQLite** database

### 📈 Analytics & Reports
- Highlights your **Best** and **Worst performing** stocks
- **Tax Calculator** — estimates STCG (15%) and LTCG (12.5% above ₹1L) liabilities based on unrealized gains
- **Export to Excel** — download a full portfolio report with a Summary sheet in one click

### 🧠 AI Assistant
- Built-in AI chat interface powered by **Google Gemini** or **OpenAI**
- Portfolio context (holdings, sectors, P&L) is automatically passed to the AI
- Ask anything: *"Which sectors am I overweight in?"*, *"What's my risk exposure?"*, *"Suggest rebalancing ideas"*
- Bring your own API key — no keys are stored

### ⚙️ Settings
- Toggle between **INR (₹)** and **USD ($)** display
- Customizable **STCG and LTCG tax rate sliders**
- **Reset Application** button to clear all holdings and cache

---

## 🖥️ Screenshots

> *Upload your broker export and the dashboard populates automatically with live data.*

| Dashboard | Analytics |
|---|---|
| Sector allocation, top holdings, portfolio heatmap | Best/worst performers, tax estimates, Excel export |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Arshad494/stock-portfolio-app.git
cd stock-portfolio-app

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 📂 Project Structure

```
stock-portfolio-app/
├── app.py               # Main Streamlit application
├── portfolio.db         # SQLite database (auto-created on first run)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical calculations |
| `plotly` | Interactive charts |
| `yfinance` | Real-time stock price data |
| `xlsxwriter` | Excel export |
| `requests` | HTTP requests |
| `google-generativeai` | Google Gemini AI integration |
| `openai` | OpenAI GPT integration |

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📥 Supported Broker Formats

The app auto-detects column names from your broker's export file.

| Broker | Format | Notes |
|---|---|---|
| **Groww** | CSV | Use the "Holdings" export |
| **Zerodha** | Excel (.xlsx) | Console holdings export |
| **Upstox** | CSV | Portfolio export |
| **Manual** | Any CSV/Excel | Columns: Ticker/Name, Quantity, Avg Price |

> **Tip:** If your broker isn't listed, any CSV/Excel with `Ticker` (or `Stock Name`), `Quantity`, and `Average Price` columns will work.

---

## 🤖 AI Assistant Setup

The AI Assistant requires an API key from your preferred provider:

**Google Gemini (Recommended — Free Tier Available)**
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Generate a free API key
3. Select **Google Gemini** in the app and paste your key

**OpenAI (GPT)**
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create an API key
3. Select **OpenAI** in the app and paste your key

> Keys are entered per-session and are never stored or logged.

---

## 🔒 Privacy & Data

- All portfolio data is stored **locally** in `portfolio.db` (SQLite)
- No data is sent to any external server (except real-time price lookups via Yahoo Finance)
- API keys for AI providers are session-only and never persisted

---

## 🗺️ Roadmap

- [ ] Historical P&L charting with date range filters
- [ ] Portfolio comparison against Nifty 50 / Sensex benchmark
- [ ] SIP tracker and goal planning module
- [ ] Dividend tracking and yield calculator
- [ ] Multi-portfolio support
- [ ] Streamlit Cloud / Docker deployment guide

---

## 🙌 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Arshad**  
GitHub: [@Arshad494](https://github.com/Arshad494)

---

<p align="center">Made with ❤️ and Python for smarter investing</p>
