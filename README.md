1. Clone or download the repository

# ⚡ Fast RAG – Groq & Gemini

**Fast document Q&A chat with optional high-accuracy Docling processing**

A lightweight, speed-optimized **Retrieval-Augmented Generation** (RAG) application built with Streamlit that lets you:

- Upload PDF, DOCX, PPTX, CSV, TXT, MD, HTML files
- Chat with your documents using **Groq (Llama 3.3 70B)** or **Google Gemini 1.5 Flash**
- Choose between **very fast** default processing or **smarter but slower** Docling mode for complex layouts & tables

## ✨ Features

- ⚡ **Two speed modes**:
  - Fast mode (default): simple loaders + recursive text splitting (~2–10 s)
  - Docling mode: advanced layout/table understanding (~30–180 s)
- 🤖 **Dual LLM support** — Groq Llama-3.3-70B or Gemini 1.5 Flash (auto-fallback)
- 🧠 **Chat history** preserved in session
- 📄 **Source citation** with file name + page + snippet preview
- 🔐 API keys via `.env` **or** Streamlit Cloud **secrets**
- 🛠️ Clean error handling & user feedback

## 🚀 Quick Start (Local)

1. Clone or download the repository

```bash
git clone https://github.com/YOUR-USERNAME/fast-rag-streamlit.git
cd fast-rag-streamlit

2. Install dependencies

```bash
pip install -r requirements.txt

Typical requirements.txt:

streamlit>=1.38
python-dotenv
langchain
langchain-core
langchain-community
langchain-huggingface
langchain-groq
langchain-google-genai
faiss-cpu
pypdf
docx2txt
unstructured
python-pptx
python-docx
langchain-docling          # only needed if using Docling mode

3. Create .env file in root folder

```bash
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GOOGLE_API_KEY=AIz...xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# optional: only one is required

4. Run the application

```bash
streamlit run app.py

📊 When to Use Which Mode

Mode,Speed,Best for,Table extraction,Layout awareness,Recommended
Fast (default),★★★★★,"Most documents, speed critical",Basic,Low,Yes ✅
Docling,★☆☆☆☆,"Complex PDFs, tables, forms, multi-column",Excellent,High,Only if needed

🛠️ Tech Stack

Frontend — Streamlit
Embeddings — all-MiniLM-L6-v2 (Hugging Face)
Vector Store — FAISS (CPU)
LLMs — Groq (Llama-3.3-70B) • Google Gemini 1.5 Flash
Document loading — PyPDF / Unstructured / Docx2txt / python-pptx + Docling (optional)
Chunking — RecursiveCharacterTextSplitter (1000/200)

⚙️ Configuration & Tips

Change model temperature, chunk size, retriever k, etc. directly in code
Want better table support without Docling? → consider adding unstructured[local-inference] + paddle/tesseract
Deploying to Streamlit Community Cloud? → add secrets in the app settings

📝 License
MIT
🙌 Acknowledgments
Built with love using:

LangChain
Groq
Google Generative AI
Docling (optional powerhouse)
FAISS

Happy RAG-ing! 🚀
