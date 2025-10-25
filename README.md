# Ultimate IHRP RAG Chatbot

## 🚀 Features
- No LangChain dependencies (pure implementation)
- RAG Fusion retrieval
- Source credibility scoring
- Guardrails validation
- Feedback system
- Cloud or local deployment

## 📦 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
playwright install chromium
```

### 2. Setup Environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run
```bash
python ultimate_ihrp_rag.py
```

## 🐳 Docker Deployment
```bash
docker-compose up -d
```

## ☁️ Cloud Deployment

### Hugging Face Spaces
1. Create new Space at huggingface.co
2. Upload all files
3. Add secrets in Settings

### Railway
1. Connect GitHub repo
2. Add environment variables
3. Deploy

## 📊 Tech Stack
- OpenAI API (GPT-4o)
- Qdrant (vector database)
- Sentence-Transformers (embeddings)
- Cross-Encoder (reranking)
- Playwright (web scraping)
- Gradio (UI)

## 🔧 Configuration
Edit `Config` class in the code to customize:
- Models
- Retrieval parameters
- Auto-update schedule
- Credibility scoring
# rag-chatbot
RAG LLM Chatbot
