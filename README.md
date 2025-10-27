# 🏥 MemorialX Healthcare Assistant

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688)
![License](https://img.shields.io/badge/license-Healthcare-green)
![Status](https://img.shields.io/badge/status-Production%20Ready-success)

**An intelligent, real-time healthcare monitoring system with AI-powered patient support**

[Features](#-features) • [Tech Stack](#-technology-stack) • [Installation](#-quick-start) • [API](#-api-endpoints)

</div>

---

## 🎯 Overview

**MemorialX Healthcare Assistant** is an enterprise-grade healthcare monitoring platform combining real-time patient tracking with AI-powered insights. Built for healthcare professionals and patients with modern web technologies.

### Key Highlights

- ⚡ Real-time monitoring via WebSocket
- 🤖 AI-powered chatbot with RAG (LangChain)
- 🧠 Smart memory system for context retention
- 📖 Gemini-powered storytelling therapy
- 📊 Comprehensive analytics dashboard
- 🎮 Interactive therapy modules
- 🌐 Multi-language (English/Arabic)
- 🔒 HIPAA-compliant architecture

---

## ✨ Features

### Healthcare Providers
- Real-time patient activity monitoring with WebSocket updates
- AI-generated insights and recommendations via Gemini
- Exportable reports (PDF/Excel)
- Multi-patient dashboard with live metrics

### Patients
- **AI Chatbot**: RAG-powered conversations with LangChain
- **Interactive Therapy**: Games, Chat, Storytelling, Relaxation, AI Art
- **Smart Memory**: Context-aware conversations with memory retention
- **Storytelling**: Gemini-powered personalized narratives
- Progress tracking and achievements
- Bilingual interface with accessibility features

---

## 🛠 Technology Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Python 3.8+, FastAPI, SQLAlchemy, Pydantic, Uvicorn |
| **Frontend** | HTML5, CSS3, JavaScript ES6+, Bootstrap 5, Chart.js |
| **AI/ML** | Google Gemini API, LangChain, RAG, Hugging Face |
| **Chatbot** | LangChain Framework, RAG (Retrieval-Augmented Generation) |
| **Memory System** | LangChain Memory, Vector Stores, Embeddings |
| **Database** | SQLite (dev), PostgreSQL (prod) |
| **Real-time** | WebSockets, Async/Await |
| **Security** | JWT, Passlib, CORS |
| **Testing** | pytest, pytest-asyncio, httpx |

---

## 🏗 Architecture

<div align="center">

```
┌─────────────────────────────────────────────────────────────┐
│                    🖥️  CLIENT LAYER                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │  📊 Dashboard │  │  👤 Patient   │  │  ⚙️  Admin    │  │
│  │   Interface   │  │   Interface   │  │    Portal     │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │ WebSocket / HTTP
┌────────────────────────────▼────────────────────────────────┐
│              ⚡ API GATEWAY - FastAPI                        │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │ 🔐 Auth (JWT) │  │  🌐 REST API  │  │ 📡 WebSocket  │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│           🤖 AI & INTELLIGENCE LAYER                         │
│                                                              │
│              ┌─────────────────────────────┐                │
│              │  🦜 LangChain RAG Chatbot   │                │
│              ├─────────────────────────────┤                │
│              │  🧠 Memory Management       │                │
│              │  📚 Vector Store & Embeddings                │
│              │  🔍 Context Retrieval       │                │
│              └─────────────────────────────┘                │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              ⚙️  BUSINESS LOGIC LAYER                        │
│                                                              │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │ 📈 Activity      │      │ 📖 Storytelling  │           │
│  │    Tracking      │      │    (Gemini AI)   │           │
│  └──────────────────┘      └──────────────────┘           │
│                                                              │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │ 💡 AI Insights   │      │ 📢 Real-time     │           │
│  │    & Reports     │      │    Broadcasting  │           │
│  └──────────────────┘      └──────────────────┘           │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│           💾 DATA PERSISTENCE LAYER                          │
│                                                              │
│  ┌──────────────────────┐    ┌──────────────────────┐      │
│  │  👥 users.db         │    │  💭 memories.db      │      │
│  │  (SQLite-WAL Mode)   │    │  (Storytelling)      │      │
│  └──────────────────────┘    └──────────────────────┘      │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              🌐 EXTERNAL SERVICES                            │
│                                                              │
│     ┌──────────────────┐         ┌──────────────────┐      │
│     │ ✨ Google Gemini │         │ 🤗 Hugging Face  │      │
│     │       AI         │         │      API         │      │
│     └──────────────────┘         └──────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

</div>

### 🔄 Data Flow

**Request Flow:**
1. 🖥️ **Client** → Sends request (WebSocket/HTTP)
2. ⚡ **API Gateway** → Authenticates & validates with JWT
3. 🤖 **AI Layer** → LangChain RAG processes with context & memory
4. ⚙️ **Services** → Executes business logic (tracking/storytelling/insights)
5. 💾 **Database** → Persists data to SQLite
6. 🌐 **External APIs** → Integrates Gemini/HuggingFace when needed
7. 📢 **Broadcasting** → Real-time updates via WebSocket to all clients

**Key Features:**
- 🔐 **Security**: JWT authentication at API layer
- 🧠 **Intelligence**: RAG-powered chatbot with memory
- ⚡ **Real-time**: WebSocket for instant updates
- 📊 **Analytics**: Comprehensive tracking and insights
- 🎨 **AI Generation**: Gemini storytelling & HuggingFace art

---

## 📁 Project Structure

```
healthcare-dashboard/
├── main.py                    # Entry point
├── config.py                  # Configuration
├── routes_dashboard.py        # Dashboard APIs
├── storytelling_feature/      # Storytelling module
├── templates/                 # HTML templates
├── static/                    # CSS, JS, images
│   └── js/activity-tracker.js # Real-time client
├── tests/                     # Test suite
└── requirements.txt           # Dependencies
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/ghreeb1/Memorixal-Healthcare-Assistant.git
cd Memorixal-Healthcare-Assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```env
SECRET_KEY=your_secret_key_min_32_chars
GEMINI_API_KEY=your_gemini_key          # Required for AI features
HF_API_KEY=your_huggingface_key         # Optional
LANGCHAIN_API_KEY=your_langchain_key    # Optional for tracing
DEBUG=True
```

### 3. Run Application

```bash
python main.py
```

**Access:**
- Dashboard: http://localhost:8000/dashboard
- API Docs: http://localhost:8000/docs

---

## 🔌 API Endpoints

### Authentication
```bash
POST /api/auth/login
```

### Dashboard
```bash
GET  /api/v1/patients/{id}/dashboard     # Get metrics
GET  /api/v1/patients/{id}/activities    # Get activities
GET  /api/v1/patients/{id}/ai-insights   # AI insights
POST /api/report                          # Generate report
```

### Real-Time Tracking
```bash
POST /api/patient-activity/start  # Start activity
POST /api/patient-activity/end    # End activity
WS   /ws/dashboard                # WebSocket updates
```

### WebSocket Example
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/dashboard');
ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log('Update:', update);
};
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run real-time demo
python demo_realtime.py
```

---

## 🚀 Deployment

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t healthcare-dashboard .
docker run -p 8000:8000 healthcare-dashboard
```

### Production Settings
```env
DEBUG=False
SECRET_KEY=production_secret_key
ALLOWED_ORIGINS=https://yourdomain.com
DATABASE_URL=postgresql://user:pass@host/db
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| WebSocket fails | Check firewall, verify port 8000 |
| Database locked | Close other instances, enable WAL mode |
| Import errors | `pip install -r requirements.txt --force-reinstall` |
| No real-time updates | Check browser console, run demo_realtime.py |

**Debug Mode:**
```bash
LOG_LEVEL=DEBUG python main.py
```

---

## 🗺 Roadmap

**Version 2.0** (Q1 2026)
- Mobile apps (iOS/Android)
- Advanced ML prediction models
- Video consultation
- Multi-tenant support

---

## 🙏 Acknowledgments

<div align="center">

### Built With Amazing Technologies

<table>
  <tr>
    <td align="center" width="140">
      <img src="https://cdn.worldvectorlogo.com/logos/fastapi.svg" width="48" height="48" alt="FastAPI" />
      <br /><b>FastAPI</b>
    </td>
    <td align="center" width="140">
      <img src="https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg" width="48" height="48" alt="Gemini" />
      <br /><b>Google Gemini</b>
    </td>
    <td align="center" width="140">
      <img src="https://avatars.githubusercontent.com/u/126733545?s=200&v=4" width="48" height="48" alt="LangChain" />
      <br /><b>LangChain</b>
    </td>
    <td align="center" width="140">
      <img src="https://cdn.worldvectorlogo.com/logos/sqlalchemy.svg" width="48" height="48" alt="SQLAlchemy" />
      <br /><b>SQLAlchemy</b>
    </td>
  </tr>
  <tr>
    <td align="center" width="140">
      <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="48" height="48" alt="Hugging Face" />
      <br /><b>Hugging Face</b>
    </td>
    <td align="center" width="140">
      <img src="https://cdn.worldvectorlogo.com/logos/python-5.svg" width="48" height="48" alt="Python" />
      <br /><b>Python</b>
    </td>
    <td align="center" width="140">
      <img src="https://www.vectorlogo.zone/logos/sqlite/sqlite-icon.svg" width="48" height="48" alt="SQLite" />
      <br /><b>SQLite</b>
    </td>
    <td align="center" width="140">
      <img src="https://cdn.worldvectorlogo.com/logos/bootstrap-4.svg" width="48" height="48" alt="Bootstrap" />
      <br /><b>Bootstrap</b>
    </td>
  </tr>
  <tr>
    <td align="center" width="140">
      <img src="https://cdn.worldvectorlogo.com/logos/websocket.svg" width="48" height="48" alt="WebSocket" />
      <br /><b>WebSocket</b>
    </td>
    <td align="center" width="140">
      <img src="https://www.chartjs.org/img/chartjs-logo.svg" width="48" height="48" alt="Chart.js" />
      <br /><b>Chart.js</b>
    </td>
    <td align="center" width="140">
      <img src="https://docs.pytest.org/en/stable/_static/pytest1.png" width="48" height="48" alt="Pytest" />
      <br /><b>Pytest</b>
    </td>
    <td align="center" width="140">
      <img src="https://cdn.worldvectorlogo.com/logos/jwt-3.svg" width="48" height="48" alt="JWT" />
      <br /><b>JWT</b>
    </td>
  </tr>
</table>

<br />

</div>

---

## 👥 Contributors

<div align="center">

### Development Team

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/ghreeb1">
        <img src="https://github.com/ghreeb1.png" width="100px;" alt="Mohamed Khaled"/>
        <br />
        <sub><b>Mohamed Khaled</b></sub>
      </a>
      <br />
      <sub>Lead Developer & AI Specialist</sub>
      <br />
      <a href="https://www.linkedin.com/in/mohamed-khaled-3a9021263">
        <img src="https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white" alt="LinkedIn"/>
      </a>
      <a href="mailto:qq11gharipqq11@gmail.com">
        <img src="https://img.shields.io/badge/-Gmail-D14836?style=flat&logo=gmail&logoColor=white" alt="Gmail"/>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/SARAELSAFTY">
        <img src="https://github.com/SARAELSAFTY.png" width="100px;" alt="Sara El Safty"/>
        <br />
        <sub><b>Sara El Safty</b></sub>
      </a>
      <br />
      <sub>Core Contributor</sub>
    </td>
    <td align="center">
      <a href="https://github.com/Iam-M-i-r-z-a">
        <img src="https://github.com/Iam-M-i-r-z-a.png" width="100px;" alt="Mirza"/>
        <br />
        <sub><b>Mirza</b></sub>
      </a>
      <br />
      <sub>Core Contributor</sub>
    </td>
  </tr>
</table>

*Building intelligent healthcare solutions together*

</div>

---

## 📧 Contact

<div align="center">

### Get In Touch

For questions, suggestions, or collaboration opportunities:

**Project Lead:** Mohamed Khaled

<p>
  <a href="mailto:qq11gharipqq11@gmail.com" target="_blank">
    <img src="https://img.shields.io/badge/-Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Gmail"/>
  </a>
  <a href="https://www.linkedin.com/in/mohamed-khaled-3a9021263" target="_blank">
    <img src="https://img.shields.io/badge/-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
  </a>
  <a href="https://github.com/ghreeb1/Memorixal-Healthcare-Assistant" target="_blank">
    <img src="https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
  </a>
</p>

</div>

---

<div align="center">

[![GitHub Issues](https://img.shields.io/github/issues/ghreeb1/Memorixal-Healthcare-Assistant)](https://github.com/ghreeb1/Memorixal-Healthcare-Assistant/issues)
[![GitHub Stars](https://img.shields.io/github/stars/ghreeb1/Memorixal-Healthcare-Assistant)](https://github.com/ghreeb1/Memorixal-Healthcare-Assistant/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/ghreeb1/Memorixal-Healthcare-Assistant)](https://github.com/ghreeb1/Memorixal-Healthcare-Assistant/network/members)

**Made with ❤️ for better healthcare**

</div>
