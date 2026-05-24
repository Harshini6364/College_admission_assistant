# 🎓 College Admission Assistant

A production-grade AI-powered chatbot that answers college admission queries by reading your college's official documents using a modular RAG (Retrieval-Augmented Generation) pipeline.

---

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [RAG Pipeline Architecture](#rag-pipeline-architecture)
- [Design Patterns](#design-patterns)
- [SOLID Principles](#solid-principles)
- [Installation and Setup](#installation-and-setup)
- [How to Use](#how-to-use)
- [Evaluation](#evaluation)
- [Credits](#credits)
- [License](#license)

---

## Project Description

### What does it do?
The College Admission Assistant is a Streamlit-based chatbot that answers questions about college admissions, fee structure, and hostel facilities. Instead of relying on general AI knowledge, it reads your college's actual PDF documents and answers strictly from them — preventing hallucination and ensuring accuracy.

### Why did we build it?
Traditional chatbots make up answers from training data. Our assistant uses RAG — it retrieves real information from college documents before generating any answer. This ensures every response is grounded in actual college policy documents.

### What problem does it solve?
Students and parents often struggle to find specific information about admissions, fees, and facilities scattered across multiple PDF documents. This assistant gives instant, accurate, document-grounded answers in a conversational format.

### What makes it stand out?
- Full 6-stage RAG pipeline (Rewrite → Retrieve → Rerank → Refine → Insert → Generate)
- CrossEncoder ML reranking for production-grade retrieval accuracy
- GOF design patterns (Singleton, Factory Method, Template Method)
- SOLID principles with low coupling and high cohesion
- Real evaluation using RAGAS metrics
- Modular industrial architecture — each file has exactly one responsibility

---

## Features

- 📄 PDF document ingestion using Docling (with PyPDFLoader fallback)
- 🔍 Hybrid retrieval — BM25 keyword search + FAISS semantic search
- 🎯 CrossEncoder ML reranking for accurate chunk selection
- 🧹 3-stage chunk refinement — filters noise before reaching LLM
- 💬 Multi-session chat with persistent history
- 📊 Real-time evaluation dashboard with ground truth matching
- 📈 Deep RAGAS evaluation — faithfulness, relevancy, precision, recall
- 🏗️ Reusable RAGPipeline class — Steel Thread end-to-end in one call
- 🔒 Singleton LLM — one instance shared across entire application

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq (llama-3.1-8b-instant) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| Keyword Search | BM25 (rank-bm25) |
| Reranker | CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| PDF Parsing | Docling / PyPDFLoader |
| Evaluation | RAGAS + word overlap scoring |
| UI | Streamlit |
| Language | Python 3.11 |

---

## Project Structure

```
college-admission-rag/
│
├── app.py                        # Entry point — UI orchestration only
├── config.py                     # All constants, env vars, model names
├── qa_pairs.json                 # Ground truth QA pairs for evaluation
├── requirements.txt              # Project dependencies
├── .env                          # API keys (not committed to GitHub)
├── .gitignore
├── README.md
│
├── core/                         # RAG pipeline — pure logic, no UI
│   ├── __init__.py
│   ├── loader.py                 # GOF Factory — document loading
│   ├── rag.py                    # Chunking, FAISS, BM25 indexing
│   ├── retriever.py              # Hybrid retrieval + CrossEncoder rerank + refine
│   ├── prompts.py                # Prompt template builder
│   ├── llm_handler.py            # GOF Singleton — LLM instance management
│   ├── pipeline.py               # GOF Template Method — Steel Thread pipeline
│   └── evaluation.py             # RAGAS + word overlap evaluation
│
├── components/                   # Streamlit UI components
│   ├── __init__.py
│   ├── sidebar.py                # Chat history sidebar
│   └── eval_panel.py             # Evaluation dashboard
│
├── utils/                        # Shared helpers
│   ├── __init__.py
│   ├── logger.py                 # Centralised logging
│   ├── chat_history.py           # Load/save chat sessions
│   └── qa_loader.py              # Ground truth loader and matcher
│
└── data/                         # College PDF documents
    ├── admission.pdf
    ├── fees.pdf
    └── hostel.pdf
```

---

## RAG Pipeline Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  STAGE 1: REWRITE                   │
│  RecursiveCharacterTextSplitter     │
│  chunk_size=1000, overlap=200       │
│  File: core/rag.py                  │
└──────────────────┬──────────────────┘
                   │
    ▼
┌─────────────────────────────────────┐
│  STAGE 2: RETRIEVE (Hybrid)         │
│  ├── BM25 keyword search            │
│  └── FAISS embedding search         │
│  File: core/retriever.py            │
└──────────────────┬──────────────────┘
                   │
    ▼
┌─────────────────────────────────────┐
│  STAGE 3: RERANK                    │
│  CrossEncoder (ms-marco-MiniLM)     │
│  Scores each (query, chunk) pair    │
│  File: core/retriever.py            │
└──────────────────┬──────────────────┘
                   │
    ▼
┌─────────────────────────────────────┐
│  STAGE 4: REFINE                    │
│  Filter low score, short, off-topic │
│  File: core/retriever.py            │
└──────────────────┬──────────────────┘
                   │
    ▼
┌─────────────────────────────────────┐
│  STAGE 5: INSERT                    │
│  Build prompt with history+context  │
│  File: core/prompts.py              │
└──────────────────┬──────────────────┘
                   │
    ▼
┌─────────────────────────────────────┐
│  STAGE 6: GENERATE                  │
│  Groq LLM (llama-3.1-8b-instant)   │
│  File: core/llm_handler.py          │
└──────────────────┬──────────────────┘
                   │
    ▼
Answer + Evaluation Dashboard
```

---

## Design Patterns

### GOF Singleton — `core/llm_handler.py`
Ensures only one `ChatGroq` instance is created and reused across the entire application. Avoids repeated API connection overhead.

```python
llm = LLMSingleton.get_instance()
```

### GOF Factory Method — `core/loader.py`
`DocumentLoaderFactory` decides which loader to use based on file extension. Adding new file type support requires one line — nothing else changes.

```python
loader = DocumentLoaderFactory.get_loader(file_path)
```

### GOF Template Method — `core/pipeline.py`
`RAGPipeline` defines the skeleton of the RAG algorithm. Each stage is an overridable method. Teammate can subclass for their own project.

```python
pipeline = RAGPipeline(data_folder="data/")
result = pipeline.run("What is the admission fee?")
```

---

## SOLID Principles

| Principle | Implementation |
|---|---|
| **S** — Single Responsibility | Each file has one job: loader loads, retriever retrieves, prompts builds prompts |
| **O** — Open/Closed | Factory registry extended without modifying existing code |
| **L** — Liskov Substitution | Docling and PyPDFLoader interchangeable — both return `list[Document]` |
| **I** — Interface Segregation | UI components only import what they need |
| **D** — Dependency Inversion | `app.py` depends on abstractions, not concrete implementations |

---

## Installation and Setup

### Prerequisites
- Python 3.11
- Groq API key (free at https://console.groq.com)
- XAMPP (for MySQL — optional, for tool calling extension)

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/college-admission-rag.git
cd college-admission-rag
```

### Step 2 — Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install PyTorch first (important — must be before other packages)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 4 — Install remaining dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Create `.env` file
Create a file named `.env` in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

### Step 6 — Add your college documents
Place your PDF files inside the `data/` folder:
```
data/
├── admission.pdf
├── fees.pdf
└── hostel.pdf
```

### Step 7 — Run the application
```bash
streamlit run app.py
```

---

## How to Use

### Running the Chatbot
1. Open your browser — Streamlit will automatically open at `http://localhost:8501`
2. Type your question in the chat input at the bottom
3. The assistant will search your documents and answer

### Example Questions
```
What is the admission fee?
What documents are required for admission?
Is WiFi available in the hostel?
What is the eligibility for CSE?
Are scholarships available?
```

### Evaluation Dashboard
After every answer, the dashboard shows:
- Retrieved document chunks used to generate the answer
- Ground truth (expected answer from `qa_pairs.json`)
- Evaluation score — High / Medium / Low

### Running RAGAS Deep Evaluation
```bash
python ragas_pipeline_test.py
```
Results are saved to `evaluation_results.json` with scores for:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

### New Chat
Click **➕ New Chat** in the sidebar to start a fresh conversation.

---

## Evaluation

Two evaluation methods are implemented:

### 1. Real-time Word Overlap Scoring
Runs after every answer in the Streamlit UI. Compares the response to the matched ground truth from `qa_pairs.json` using word overlap ratio.

| Score | Meaning |
|---|---|
| High | ≥ 60% word overlap with ground truth |
| Medium | 30–60% word overlap |
| Low | < 30% overlap or no context found |

### 2. RAGAS Deep Evaluation
Run separately for comprehensive metrics:

| Metric | What it measures |
|---|---|
| Faithfulness | Is the answer grounded in retrieved context? |
| Answer Relevancy | Is the answer relevant to the question? |
| Context Precision | Are retrieved chunks actually useful? |
| Context Recall | Did retrieval find all necessary information? |

---

## Credits

**Developed by:**
- Harsh (Person 1) — GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- Bhashini (Person 2) — GitHub: [@TEAMMATE_USERNAME](https://github.com/TEAMMATE_USERNAME)

**Guided by:**
- Sriram Bharadwaj — AI/ML Mentor

**References:**
- [LangChain Documentation](https://docs.langchain.com)
- [RAGAS Documentation](https://docs.ragas.io)
- [Docling by IBM](https://github.com/docling-project/docling)
- [Groq API](https://console.groq.com)
- [sentence-transformers CrossEncoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html)

---

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2026 Harsh and Bhashini

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```
