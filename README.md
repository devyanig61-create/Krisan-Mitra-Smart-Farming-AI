# 🌱 KisanMitra — Smart Farming AI Agent

**KisanMitra** (किसान मित्र) is an AI-powered Smart Farming Advice Agent built with **Python Flask**, **IBM watsonx.ai**, **IBM Granite-4.0-8B-Instruct**, and **RAG (Retrieval-Augmented Generation)**. It provides Indian farmers with personalized guidance in **English, Hindi, and Marathi**.

---

## 🏗 Architecture

```
smart-farming-ai/
├── run.py                    # WSGI entry point
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── app/
│   ├── __init__.py           # Flask app factory
│   ├── watsonx_client.py     # IBM Granite-4.0-8B-Instruct client
│   ├── rag_engine.py         # FAISS vector store + knowledge base
│   ├── prompts.py            # Multilingual system prompts
│   └── routes/
│       ├── main.py           # Dashboard route
│       ├── chat.py           # Chatbot API (RAG + watsonx)
│       ├── weather.py        # Open-Meteo weather API
│       ├── crops.py          # Crop recommendations + disease diagnosis
│       ├── market.py         # Mandi prices + market advice
│       └── advisory.py       # Irrigation, seasonal, government schemes
└── templates/
    └── index.html            # Full responsive dashboard + chatbot UI
```

---

## ✨ Features

| Feature | Description |
|---|---|
| **AI Chatbot** | Multilingual (EN/HI/MR) farming conversations powered by IBM Granite |
| **RAG Knowledge Base** | FAISS-indexed farming knowledge for accurate, grounded answers |
| **Crop Advisor** | Personalized recommendations by soil type, season, location, water |
| **Disease Diagnosis** | Identify diseases/pests from symptoms with treatment plans |
| **Fertilizer Scheduler** | Soil-test-based NPK + micronutrient schedules |
| **Weather Forecast** | 7-day forecast with farm-specific alerts (Open-Meteo API) |
| **Mandi Prices** | Current prices from 9+ crops, 4+ mandis each; MSP comparison |
| **Market Advice** | AI guidance on when/where to sell for maximum profit |
| **Irrigation Planner** | Stage-wise irrigation schedules for all major crops |
| **Seasonal Advisory** | Season-specific, region-specific farming calendars |
| **Government Schemes** | PM-KISAN, PMFBY, PMKSY, KCC, e-NAM, Soil Health Card info |

---

## 🚀 Quick Start

### 1. Clone / Download

```bash
git clone <repo-url>
cd smart-farming-ai
```

### 2. Create virtual environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure credentials — two ways

#### Option A — Interactive setup script (recommended)
```bash
python setup_credentials.py
```
The script walks you through each credential, validates the connection against IBM watsonx.ai live, and writes `.env` automatically.

#### Option B — Manual
```bash
cp .env.example .env
```
Open `.env` and replace **all three** placeholder values:

```env
WATSONX_API_KEY=<your IBM Cloud API key>
WATSONX_PROJECT_ID=<your watsonx.ai project ID>
WATSONX_URL=https://us-south.ml.cloud.ibm.com
FLASK_SECRET_KEY=<any random string>
```

**Where to get each value:**

| Value | Steps |
|---|---|
| `WATSONX_API_KEY` | [cloud.ibm.com/iam/apikeys](https://cloud.ibm.com/iam/apikeys) → **Create** → copy immediately (shown once) |
| `WATSONX_PROJECT_ID` | [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com) → open your project → **Manage** tab → copy UUID |
| `WATSONX_URL` | Keep default `https://us-south.ml.cloud.ibm.com` unless your instance is in EU/Tokyo |

> **IBM Cloud Lite (free) account**: [cloud.ibm.com](https://cloud.ibm.com) → provision **watsonx.ai** → create a project.

### 5. Run the application

```bash
python run.py
```

Open **http://localhost:5000** in your browser.

---

## 🌐 API Endpoints

### Chat
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/chat/` | Send a message, get AI response |
| GET  | `/api/chat/history/<id>` | Retrieve conversation history |
| DELETE | `/api/chat/clear/<id>` | Clear session history |

### Weather
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/weather/?location=Pune` | 7-day forecast + farm alerts |

### Crops
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/crops/recommend` | AI crop recommendations |
| POST | `/api/crops/disease` | Disease & pest diagnosis |
| POST | `/api/crops/fertilizer` | Fertilizer schedule |
| GET  | `/api/crops/list` | All supported crops |

### Market
| Method | Endpoint | Description |
|---|---|---|
| GET  | `/api/market/prices?crop=wheat` | Mandi prices |
| POST | `/api/market/advice` | AI market selling advice |

### Advisory
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/advisory/irrigation` | Irrigation schedule |
| POST | `/api/advisory/seasonal` | Seasonal advisory |
| GET  | `/api/advisory/schemes` | Government schemes list |
| POST | `/api/advisory/schemes/advice` | Personalized scheme advice |

---

## 💬 Multilingual Chat Example

**English:**
> "What fertilizer should I use for wheat on alluvial soil?"

**Hindi:**
> "मेरी काली मिट्टी में कपास के लिए कौन सी खाद डालें?"

**Marathi:**
> "माझ्या जमिनीत सोयाबीन लावण्यासाठी योग्य खत कोणते?"

---

## 🔧 IBM watsonx.ai Setup

1. Sign up at [cloud.ibm.com](https://cloud.ibm.com) (Free Lite plan available)
2. Create a **watsonx.ai** service instance
3. Go to **Projects** → create a new project
4. Copy the **Project ID** from project settings
5. Create an **API Key** at `cloud.ibm.com/iam/apikeys`
6. Add to your `.env` file

### Model Used
- **`ibm/granite-4-h-small`** — IBM Granite-4.0 Instruct (Hybrid Small)
- Context window: 8K tokens
- Capabilities: RAG, instruction following, multilingual

---

## 📊 RAG Knowledge Base

The embedded FAISS knowledge base covers:
- **8 major crops** (wheat, rice, cotton, sugarcane, maize, soybean, tomato, mustard)
- **4 soil types** (alluvial, black, red, laterite)
- **5 diseases/pests** with treatment protocols
- **3 fertilizer guides** (NPK, organic, micronutrients)
- **3 irrigation methods** (drip, sprinkler, scheduling)
- **3 seasons** (Kharif, Rabi, Zaid)
- **2 government schemes** (PMFBY, PM-KISAN)
- **Mandi & MSP 2024-25 data**

---

## 🌦 Weather Integration

Uses **Open-Meteo API** — completely free, no API key required:
- Real-time current conditions
- 7-day forecast (temperature, rainfall, wind, UV)
- Geocoding via Nominatim (OpenStreetMap)
- Automatic farm alerts (heavy rain, heat wave, cold wave, strong winds)

---

## 📱 Dashboard Features

- **Dashboard** — weather stats, farm alerts, price summary, quick-ask
- **AI Chatbot** — full conversation with farmer profile context
- **Weather** — location-based forecast with farm alerts
- **Crop Advisor** — recommendations, disease diagnosis, fertilizer planner
- **Mandi Prices** — 9 crops, live-style pricing table with trends
- **Advisories** — irrigation schedules, seasonal calendars
- **Gov. Schemes** — all major schemes with links + AI guidance
- **Multilingual** — toggle EN / HI / MR at any time

---

## 🛠 Technology Stack

| Component | Technology |
|---|---|
| **AI Model** | IBM Granite-4.0-8B-Instruct via watsonx.ai |
| **RAG Engine** | FAISS + sentence-transformers (all-MiniLM-L6-v2) |
| **Backend** | Python Flask 3.0 |
| **Weather API** | Open-Meteo (free, no key) |
| **Market Data** | Agmarknet/e-NAM indicative pricing |
| **Frontend** | Vanilla HTML5 + CSS + JS (no framework) |
| **Languages** | English, Hindi, Marathi |

---

## 📄 License

MIT License — Free to use for agricultural and educational purposes.
