# 📚 RAG Q&A System with FAISS

A powerful Retrieval-Augmented Generation (RAG) application that lets you upload documents and ask questions about them using AI.

## 🎯 Features

- **Multi-format support**: PDF, CSV, Word, Excel, HTML, TXT, Markdown
- **Fast inference**: Powered by Groq's ultra-fast LLM API
- **Semantic search**: Uses HuggingFace embeddings and FAISS vector indexing
- **Chat interface**: Beautiful Streamlit UI with chat history
- **Source attribution**: See which document chunks your answers come from
- **Easy to use**: Upload documents → Process → Ask questions

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Groq API Key

1. Go to [console.groq.com](https://console.groq.com/keys)
2. Sign up for free
3. Create an API key
4. Add it to `.env` file:

```bash
GROQ_API_KEY=your_key_here
```

### 3. Run the App

```bash
streamlit run rag_faiss.py
```

The app will open at `http://localhost:8501`

## 📖 How to Use

1. **Upload Documents** - Click "Upload files" in the sidebar
2. **Process Documents** - Click "Process Documents" button
3. **Ask Questions** - Type your question in the chat input
4. **View Sources** - Expand the "Sources" section to see relevant chunks

## 🏗️ How It Works

```
Documents → Split into chunks → Convert to embeddings
    ↓
FAISS Index (fast similarity search)
    ↓
User Question → Find similar chunks → Groq LLM → Answer
```

## 📋 Supported File Formats

| Format | Extension |
|--------|-----------|
| PDF | `.pdf` |
| CSV | `.csv` |
| Word | `.docx` |
| Excel | `.xlsx` |
| HTML | `.html` |
| Text | `.txt` |
| Markdown | `.md` |

## ⚙️ Configuration

Edit `rag_faiss.py` to customize:

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embeddings model
CHAT_MODEL = "llama-3.1-8b-instant"  # LLM model
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 300  # Overlap between chunks
```

## 🔐 Security

- API keys are loaded from `.env` file (ignored in git)
- Never commit `.env` to version control
- For Streamlit Cloud, use Streamlit Secrets

## 📦 Tech Stack

- **Framework**: Streamlit (UI)
- **LLM**: Groq (fast inference)
- **Embeddings**: HuggingFace Sentence Transformers
- **Vector Search**: FAISS
- **Document Processing**: PyPDF, python-docx, pandas, beautifulsoup4
- **LLM Orchestration**: LangChain

## 🚢 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `GROQ_API_KEY` in Streamlit Secrets (Settings → Secrets)

### Docker

```bash
docker build -t rag-faiss .
docker run -p 8501:8501 -e GROQ_API_KEY=your_key rag-faiss
```

### Local Server

```bash
streamlit run rag_faiss.py --server.port 8501 --server.address 0.0.0.0
```

## 🛠️ Troubleshooting

### "API Key not found"
- Make sure `.env` file exists in the same folder as `rag_faiss.py`
- Check that `GROQ_API_KEY` is set correctly

### "Module not found"
- Run: `pip install -r requirements.txt`

### Slow performance
- Increase `CHUNK_SIZE` for fewer chunks
- Reduce number of documents
- Check available RAM

### CUDA/GPU issues
- The app defaults to CPU (good for macOS, Linux, Windows)
- If you have CUDA, edit the code to use GPU in `load_embeddings()`

## 📚 Learn More

- [Groq API Docs](https://console.groq.com/docs)
- [Streamlit Docs](https://docs.streamlit.io)
- [LangChain Docs](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)

## 📝 License

MIT License - Feel free to use and modify

## 🤝 Contributing

Improvements welcome! Feel free to:
- Open issues for bugs
- Submit pull requests for features
- Share feedback and suggestions

## 📧 Support

If you encounter issues:
1. Check the Troubleshooting section above
2. Make sure dependencies are installed: `pip install -r requirements.txt`
3. Verify `.env` file has correct API key
4. Check Groq API status: [status.groq.com](https://status.groq.com)

---

**Happy RAG-ing! 🎉**

Built with ❤️ for document Q&A enthusiasts
