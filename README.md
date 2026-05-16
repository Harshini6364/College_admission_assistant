# College Admission Assistant

A Hybrid Retrieval-Augmented Generation (RAG) based chatbot designed to answer college-related queries using uploaded documents. The application supports PDF and text document ingestion, hybrid retrieval using BM25 and FAISS, OCR for scanned PDFs, conversational memory, and a Streamlit-based interactive interface.

---

# Features

* PDF and Text Document Loading
* Hybrid Retrieval System

  * BM25 Keyword Search
  * FAISS Vector Similarity Search
* Recursive Chunking
* HuggingFace Embeddings
* Groq LLM Integration
* OCR Support for Scanned PDFs
* Multi-Chat Session History
* Streamlit Interactive UI
* Conversational Memory
* Hallucination Control Prompting
* RAGAS Integration Setup
* Persistent Chat History using JSON Storage

---

# Project Architecture

![UML Diagram](./images/uml.png)

```text
User Query
    ↓
Hybrid Retrieval
(BM25 + FAISS)
    ↓
Relevant Chunks
    ↓
Prompt Engineering
    ↓
Groq LLM
    ↓
Generated Response
    ↓
Evaluation / RAGAS Pipeline
```

---

# Project Structure

```text
college-admission-assistant/
│
├── data/                      # Uploaded documents
│
├── app.py                     # Main Streamlit application
├── loader.py                  # Document loading + OCR
├── rag.py                     # Chunking + embeddings + FAISS + BM25
├── retriever.py               # Hybrid retrieval logic
├── prompts.py                 # Prompt template
├── llm_handler.py             # LLM response generation
├── chat_history.py            # Persistent chat history storage
├── config.py                  # Configurations and model names
├── test_ragas.py              # RAGAS installation test
├── requirements.txt           # Project dependencies
│
├── chat_history.json          # Stored conversations
└── README.md
```

---

# Technologies Used

| Technology             | Purpose                  |
| ---------------------- | ------------------------ |
| Python                 | Backend Development      |
| Streamlit              | Web Interface            |
| LangChain              | RAG Pipeline             |
| Groq API               | LLM Inference            |
| FAISS                  | Vector Search            |
| BM25                   | Keyword Retrieval        |
| HuggingFace Embeddings | Text Embeddings          |
| PyPDFLoader            | PDF Parsing              |
| Tesseract OCR          | Scanned PDF OCR          |
| RAGAS                  | RAG Evaluation Framework |

---

# How the System Works

## 1. Document Loading

The system loads:

* PDF files
* Text files

using:

* `PyPDFLoader`
* `TextLoader`

If a PDF is scanned and contains little extractable text, OCR is automatically applied using Tesseract.

---

## 2. Chunking

Documents are split using:

```python
RecursiveCharacterTextSplitter
```

Configuration:

```python
chunk_size = 1000
chunk_overlap = 200
```

This improves retrieval quality and preserves context continuity.

---

## 3. Embedding Generation

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to convert chunks into vector embeddings.

---

## 4. Hybrid Retrieval

The retrieval pipeline combines:

### BM25

* Keyword-based sparse retrieval

### FAISS

* Dense vector similarity retrieval

Both results are merged and deduplicated for better accuracy.

---

## 5. Prompt Engineering

The prompt enforces strict rules:

* Answer only from context
* Avoid hallucinations
* Return fallback message if answer unavailable

Example:

```text
"The information is not available in the uploaded documents."
```

---

## 6. Response Generation

The retrieved context and user query are passed to:

```text
llama-3.1-8b-instant
```

through Groq API for fast inference.

---

## 7. Conversational Memory

The chatbot maintains:

* multiple chat sessions
* chat persistence
* memory-based conversations

using:

* Streamlit Session State
* ConversationBufferMemory

---

# Persistent Chat History

Initially, chat history was stored using:

```python
st.session_state
```

`st.session_state` stores data only in RAM memory.

This means:

* data exists only while the app is running
* refreshing the browser may reset sessions
* restarting Streamlit removes history

To solve this, persistent storage was implemented using:

```text
chat_history.json
```

All conversations are automatically saved into the JSON file.

Example:

```json
{
    "Admission Process": [
        ["user", "tell me about admissions"],
        ["assistant", "Admissions start in June"]
    ]
}
```

This ensures:

* chat history remains after app restart
* conversations are permanently stored
* multiple chat sessions are preserved

This mimics how production-grade conversational AI systems maintain user conversations.

---

# RAGAS Integration

The project includes initial integration of:

RAGAS (Retrieval-Augmented Generation Assessment)

Current capabilities:

* Evaluation pipeline setup
* Dataset preparation
* Retrieval transparency
* Answer quality analysis preparation

Future scope:

* Faithfulness scoring
* Context precision metrics
* Automated evaluation scoring

---

# Installation

## 1. Clone Repository

```bash
git clone <your-repository-url>
```

---

## 2. Navigate to Project

```bash
cd college-admission-assistant
```

---

## 3. Create Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

# Run the Application

```bash
streamlit run app.py
```

---

# Testing RAGAS

Run:

```bash
python test_ragas.py
```

Expected Output:

```text
RAGAS successfully installed
```

---

# Application Workflow

1. Upload documents into `data/`
2. Start Streamlit app
3. Ask college-related questions
4. Hybrid retriever fetches relevant chunks
5. LLM generates grounded response

---

# Example Questions

* What is the admission process?
* What documents are required?
* What is the hostel fee?
* Is WiFi available in hostels?
* What is the eligibility for CSE?

---

# Hallucination Prevention

The assistant is designed to:

* answer only from retrieved context
* avoid unsupported information
* return fallback response when information is missing

This improves reliability and trustworthiness.

---

# Future Improvements

* Semantic Chunking
* Reranking
* Metadata Filtering
* Full RAGAS Metric Evaluation
* Agentic RAG
* Corrective RAG
* Vector Database Scaling
* Analytics Dashboard

---

# Author

Harshini

---

# License

This project is developed for educational and learning purposes.
