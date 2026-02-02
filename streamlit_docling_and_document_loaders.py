# ───────────────────────────────────────────────────────
#   IMPORTANT: Put this at the VERY TOP to fix tokenizer warning
# ───────────────────────────────────────────────────────
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Simple fast loaders
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader
)

# DOCLING (optional - only if user enables it)
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

# ───────────────────────────────────────────────────────
# Streamlit Page Setup
# ───────────────────────────────────────────────────────
st.set_page_config(page_title="Fast RAG with Optional Docling", layout="wide")
st.title("⚡ Fast RAG (Groq & Gemini)")
st.markdown("**Fast by default** | Optional Docling for complex documents")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ───────────────────────────────────────────────────────
# Load API keys (Support for .env AND Streamlit Secrets)
# ───────────────────────────────────────────────────────
# Local development uses .env, Streamlit Cloud uses st.secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

# ───────────────────────────────────────────────────────
# FAST document loading (simple loaders)
# ───────────────────────────────────────────────────────
def get_simple_loader(file_path: str, extension: str):
    """Returns fast, simple loader based on file type"""
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".md": UnstructuredMarkdownLoader,
        ".docx": Docx2txtLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".html": UnstructuredHTMLLoader,
    }
    loader_class = loaders.get(extension.lower())
    return loader_class(file_path) if loader_class else None

def process_files_fast(uploaded_files):
    """Fast processing with simple loaders + text splitter"""
    all_docs = []
    temp_dir = tempfile.mkdtemp()
    
    # Text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    for file in uploaded_files:
        ext = Path(file.name).suffix.lower()
        temp_path = os.path.join(temp_dir, file.name)

        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        try:
            # Get simple loader
            loader = get_simple_loader(temp_path, ext)
            
            if loader:
                # Load document
                docs = loader.load()
                
                # Split into chunks
                chunks = text_splitter.split_documents(docs)
                
                # Add source metadata
                for chunk in chunks:
                    chunk.metadata["source_file"] = file.name
                
                all_docs.extend(chunks)
                st.success(f"✅ {file.name}: {len(chunks)} chunks")
            else:
                st.warning(f"⚠️ {file.name}: Unsupported format")
            
        except Exception as e:
            st.error(f"❌ {file.name}: {str(e)}")

    return all_docs

def process_files_with_docling(uploaded_files):
    """Slower but smarter processing with Docling"""
    all_docs = []
    temp_dir = tempfile.mkdtemp()

    for file in uploaded_files:
        ext = Path(file.name).suffix.lower()
        temp_path = os.path.join(temp_dir, file.name)

        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        try:
            loader = DoclingLoader(
                file_path=temp_path,
                export_type=ExportType.DOC_CHUNKS
            )
            docs = loader.load()
            
            # Add source metadata
            for doc in docs:
                doc.metadata["source_file"] = file.name
            
            # Filter empty docs
            docs = [d for d in docs if d.page_content and d.page_content.strip()]
            
            if docs:
                all_docs.extend(docs)
                st.success(f"✅ {file.name}: {len(docs)} semantic chunks")
            else:
                st.warning(f"⚠️ {file.name}: No content extracted")
            
        except Exception as e:
            st.error(f"❌ {file.name}: {str(e)}")

    return all_docs

# ───────────────────────────────────────────────────────
# LLM Setup
# ───────────────────────────────────────────────────────
def get_llm():
    if GROQ_API_KEY:
        try:
            return ChatGroq(
                model_name="llama-3.3-70b-versatile",
                groq_api_key=GROQ_API_KEY,
                temperature=0.3
            )
        except Exception as e:
            st.warning(f"⚠️ Groq failed: {e}")
    
    if GOOGLE_API_KEY:
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3
            )
        except Exception as e:
            st.error(f"❌ Gemini failed: {e}")
    
    return None

# ───────────────────────────────────────────────────────
# Sidebar
# ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 API Keys")
    
    if GROQ_API_KEY:
        st.success("✅ Groq loaded")
    else:
        st.warning("⚠️ No Groq key")
    
    if GOOGLE_API_KEY:
        st.success("✅ Gemini loaded")
    else:
        st.warning("⚠️ No Gemini key")
    
    st.divider()
    
    # PROCESSING MODE SELECTOR
    st.header("⚙️ Processing Mode")
    use_docling = st.checkbox(
        "Use Docling (slower, smarter)",
        value=False,
        help="Docling: Better for complex PDFs with tables/layouts but MUCH slower. Default: Fast simple loaders"
    )
    
    if use_docling:
        st.warning("🐌 Docling mode: Slower but better for complex docs")
    else:
        st.info("⚡ Fast mode: Simple loaders + text splitting")
    
    st.divider()
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, PPTX, HTML, CSV, TXT, MD",
        type=["pdf", "docx", "pptx", "html", "csv", "txt", "md"],
        accept_multiple_files=True
    )

    if st.button("Process Documents", type="primary"):
        if not (GROQ_API_KEY or GOOGLE_API_KEY):
            st.error("❌ Add API keys to .env!")
        elif not uploaded_files:
            st.warning("⚠️ Upload files first")
        else:
            # Choose processing method
            if use_docling:
                with st.spinner("🔄 Docling processing (this may take a while)..."):
                    raw_docs = process_files_with_docling(uploaded_files)
            else:
                with st.spinner("⚡ Fast processing..."):
                    raw_docs = process_files_fast(uploaded_files)
            
            if not raw_docs:
                st.error("❌ No content extracted")
                st.stop()
            
            st.info(f"📄 Total: {len(raw_docs)} chunks")
            
            with st.spinner("🔄 Creating embeddings..."):
                try:
                    embeddings = HuggingFaceEmbeddings(
                        model_name="all-MiniLM-L6-v2",
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": True}
                    )
                    
                    st.session_state.vectorstore = FAISS.from_documents(
                        documents=raw_docs,
                        embedding=embeddings
                    )
                    
                    st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": 4}
                    )
                    
                    st.success(f"✅ Ready! {len(raw_docs)} chunks indexed")
                    
                except Exception as e:
                    st.error(f"❌ Indexing failed: {str(e)}")

    st.markdown("---")
    if st.session_state.get("chat_history"):
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ───────────────────────────────────────────────────────
# Main Chat Interface
# ───────────────────────────────────────────────────────
if st.session_state.vectorstore and st.session_state.retriever:
    llm = get_llm()
    
    if not llm:
        st.error("❌ No LLM available")
    else:
        if GROQ_API_KEY:
            st.info("🤖 Groq (Llama 3.3 70B)")
        else:
            st.info("🤖 Gemini (1.5 Flash)")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer based on context. If context doesn't contain the answer, say so."),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
            ("system", "Context:\n{context}")
        ])

        generation_chain = (
            RunnableParallel({
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
                "context": lambda x: "\n\n".join([doc.page_content for doc in x["docs"]]),
            })
            | prompt
            | llm
            | StrOutputParser()
        )

        for msg in st.session_state.chat_history:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)

        if user_query := st.chat_input("Ask about your documents..."):
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        retrieved_docs = st.session_state.retriever.invoke(user_query)
                        
                        answer = generation_chain.invoke({
                            "question": user_query,
                            "chat_history": st.session_state.chat_history,
                            "docs": retrieved_docs
                        })

                        st.markdown(answer)

                        st.markdown("**Sources:**")
                        if retrieved_docs:
                            unique_docs = []
                            seen = set()
                            for doc in retrieved_docs:
                                fname = doc.metadata.get("source_file", "Unknown")
                                page = doc.metadata.get("page", "N/A")
                                key = (fname, str(page))
                                if key not in seen:
                                    seen.add(key)
                                    unique_docs.append(doc)

                            for i, doc in enumerate(unique_docs, 1):
                                filename = doc.metadata.get("source_file", "Unknown")
                                page = doc.metadata.get("page", "N/A")
                                snippet = doc.page_content.strip()[:180].replace("\n", " ") + "..."
                                st.caption(f"[{i}] **{filename}** • Page {page}\n{snippet}")

                        st.session_state.chat_history.append(HumanMessage(content=user_query))
                        st.session_state.chat_history.append(AIMessage(content=answer))
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

else:
    st.info("👈 Upload files and click **Process Documents**")
    
    with st.expander("📖 Speed Comparison"):
        st.markdown("""
        ### ⚡ Fast Mode (Default - Recommended)
        **Speed:** ~2-5 seconds for typical PDFs
        - Uses PyPDFLoader, Docx2txtLoader, etc.
        - Simple text extraction + chunking
        - Perfect for most use cases
        - **Use when:** Standard documents, speed matters
        
        ### 🐌 Docling Mode (Optional)
        **Speed:** ~30-120 seconds for same PDFs
        - Deep document analysis
        - Table extraction, layout detection
        - Semantic chunking
        - **Use when:** Complex PDFs with tables/forms, accuracy > speed
        
        ### 💡 Recommendation:
        Start with **Fast Mode**. Only enable Docling if:
        - Your PDFs have complex tables
        - You need precise layout understanding
        - Speed is not critical
        """)