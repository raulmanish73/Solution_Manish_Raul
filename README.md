# Adobe AI Leadership Insight & Multimodal Agentic RAG 🚀

An end-to-end **Multimodal Agentic Retrieval-Augmented Generation (RAG)** system designed to analyze Adobe’s financial reports and leadership communications. This project demonstrates two production-grade multimodal agent architectures capable of answering complex financial queries grounded in **text, tables, charts, and images extracted from PDFs**.

The system evolves from an initial **CLIP + BGE hybrid agent** to a fully optimized **Nomic Vision–based multimodal embedding agent**, improving retrieval speed, scalability, and architectural simplicity.

---

# 🌟 Overview

Financial documents contain mission-critical insights embedded in:

- Text narratives
- Financial tables
- Charts and graphs
- Visual performance summaries

Traditional text-only RAG systems fail to capture these multimodal signals.

This project implements **Agentic Multimodal RAG using LangGraph**, enabling the agent to:

- Understand both text and visual financial data
- Retrieve relevant multimodal context
- Generate grounded financial answers
- Provide source attribution for traceability

---

# 🧠 Solution Evolution

This repository contains **two complete multimodal agent architectures**:

## Solution 1: Hybrid Multimodal Agent (CLIP + BGE + GPT-4o)

This was the initial architecture designed for high-precision financial analysis.

### Architecture

Stateful agent graph with specialized nodes:

1. Guardrail Node  
   Filters queries to ensure relevance to Adobe’s business domain.

2. Retriever Node  
   Hybrid retrieval using:
   - BGE-Large for text retrieval
   - CLIP embeddings for image/chart association

3. Reranker Node  
   Uses LLM pointwise scoring to filter irrelevant financial context.

4. Multimodal Answer Generator  
   Uses GPT-4o Vision to synthesize answers from text and image context.

5. Reflector Node  
   Evaluates answer quality and re-triggers retrieval if confidence is low.

---

### Key Characteristics

- Hybrid multimodal retrieval
- LLM-based reranking
- Reflection loop for answer correction
- Cloud-based embeddings and reasoning

---

### Technical Stack

- Framework: LangChain, LangGraph  
- Text Embeddings: BAAI/bge-large-en-v1.5  
- Image Embeddings: OpenAI CLIP (ViT-B/32)  
- LLM: GPT-4o (Vision + Reasoning)  
- Vector Store: FAISS  
- Document Processing: Unstructured (Hi-Res)

---

### Important Files

- `leadership_insight_agent.ipynb` – Main hybrid multimodal agent  
- `reranking_expt_notebook.ipynb` – Reranking experiments  
- `config.py` – Configurable parameters  

---

---

## Solution 2: Production Multimodal Agent (Nomic Vision + FAISS + LangGraph)

This is the optimized production architecture designed for:

- Faster retrieval
- Fully local embeddings
- Reduced architectural complexity
- Improved scalability

---

# 🧠 System Architecture (Production Version)

## Offline Pipeline
PDF → Extract → Embed → Save FAISS Index + Metadata

## Online Agent Pipeline
User Query → Agent → Retrieve → Multimodal LLM → Answer + Sources


---

# ⚙️ Core Components

## Document Extraction

Extracts multimodal content:

- Text chunks
- Tables
- Images
- Page renderings

Each item stored as structured metadata:

json
{
  "page": 0,
  "type": "text",
  "text": "...",
  "path": "extracted_data/text/file.txt"
}


## Multimodal Embeddings

This solution uses: nomic-ai/nomic-embed-vision-v1

This model generates unified embeddings for:

- Text
- Images
- Tables
- Page images

All modalities are embedded into the same semantic vector space, enabling direct similarity comparison between text and visual content.

This eliminates the need for separate embedding pipelines and improves multimodal retrieval accuracy.

---

## Vector Store (Persistent FAISS)

Embeddings are stored in a persistent FAISS index:
vector_store/
├── index.faiss
├── metadata.pkl
└── config.json


Benefits:

- Sub-second retrieval latency
- No need to regenerate embeddings at runtime
- Scalable to millions of documents
- Optimized for local and production environments

---

## Retriever Node

The Retriever Node performs similarity search using FAISS.

Input: User Query
Process: Query → Multimodal Embedding → FAISS Search → Top-K Results
Output: Retrieved multimodal documents


Retrieved documents may include:

- Text chunks
- Tables
- Images
- Page renderings

---

### Answer Node

The Answer Node synthesizes the final response using:

- Retrieved text context
- Retrieved image context
- System prompt grounding
- User query


### ⚡ Quick Start

##### Clone the Repository

git clone https://github.com/raulmanish73/Solution_Manish_Raul.git

cd Solution_Manish_Raul

##### Create Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate

##### Install Dependencies

pip install -r requirements.txt

##### Run Jupter notebook

final_solution_with_new_approach.ipynb
leadership_insight_agent.ipynb

# Adobe AI Leadership Insight & Multimodal Agentic RAG 🚀

An end-to-end **Multimodal Agentic Retrieval-Augmented Generation (RAG)** system designed to analyze Adobe’s financial reports and leadership communications. This project demonstrates two production-grade multimodal agent architectures capable of answering complex financial queries grounded in **text, tables, charts, and images extracted from PDFs**.

The system evolves from an initial **CLIP + BGE hybrid agent** to a fully optimized **Nomic Vision–based multimodal embedding agent**, improving retrieval speed, scalability, and architectural simplicity.

---

# 🌟 Overview

Financial documents contain mission-critical insights embedded in:

- Text narratives
- Financial tables
- Charts and graphs
- Visual performance summaries

Traditional text-only RAG systems fail to capture these multimodal signals.

This project implements **Agentic Multimodal RAG using LangGraph**, enabling the agent to:

- Understand both text and visual financial data
- Retrieve relevant multimodal context
- Generate grounded financial answers
- Provide source attribution for traceability

---

# 🧠 Solution Evolution

This repository contains **two complete multimodal agent architectures**:

## Solution 1: Hybrid Multimodal Agent (CLIP + BGE + GPT-4o)

This was the initial architecture designed for high-precision financial analysis.

### Architecture

Stateful agent graph with specialized nodes:

1. Guardrail Node  
   Filters queries to ensure relevance to Adobe’s business domain.

2. Retriever Node  
   Hybrid retrieval using:
   - BGE-Large for text retrieval
   - CLIP embeddings for image/chart association

3. Reranker Node  
   Uses LLM pointwise scoring to filter irrelevant financial context.

4. Multimodal Answer Generator  
   Uses GPT-4o Vision to synthesize answers from text and image context.

5. Reflector Node  
   Evaluates answer quality and re-triggers retrieval if confidence is low.

---

### Key Characteristics

- Hybrid multimodal retrieval
- LLM-based reranking
- Reflection loop for answer correction
- Cloud-based embeddings and reasoning

---

### Technical Stack

- Framework: LangChain, LangGraph  
- Text Embeddings: BAAI/bge-large-en-v1.5  
- Image Embeddings: OpenAI CLIP (ViT-B/32)  
- LLM: GPT-4o (Vision + Reasoning)  
- Vector Store: FAISS  
- Document Processing: Unstructured (Hi-Res)

---

### Important Files

- `leadership_insight_agent.ipynb` – Main hybrid multimodal agent  
- `reranking_expt_notebook.ipynb` – Reranking experiments  
- `config.py` – Configurable parameters  

---

---

## Solution 2: Production Multimodal Agent (Nomic Vision + FAISS + LangGraph)

This is the optimized production architecture designed for:

- Faster retrieval
- Fully local embeddings
- Reduced architectural complexity
- Improved scalability

---

# 🧠 System Architecture (Production Version)

## Offline Pipeline
PDF → Extract → Embed → Save FAISS Index + Metadata

## Online Agent Pipeline
User Query → Agent → Retrieve → Multimodal LLM → Answer + Sources


---

# ⚙️ Core Components

## Document Extraction

Extracts multimodal content:

- Text chunks
- Tables
- Images
- Page renderings

Each item stored as structured metadata:

json
{
  "page": 0,
  "type": "text",
  "text": "...",
  "path": "extracted_data/text/file.txt"
}


## Multimodal Embeddings

This solution uses: nomic-ai/nomic-embed-vision-v1

This model generates unified embeddings for:

- Text
- Images
- Tables
- Page images

All modalities are embedded into the same semantic vector space, enabling direct similarity comparison between text and visual content.

This eliminates the need for separate embedding pipelines and improves multimodal retrieval accuracy.

---

## Vector Store (Persistent FAISS)

Embeddings are stored in a persistent FAISS index:
vector_store/
├── index.faiss
├── metadata.pkl
└── config.json


Benefits:

- Sub-second retrieval latency
- No need to regenerate embeddings at runtime
- Scalable to millions of documents
- Optimized for local and production environments

---

## Retriever Node

The Retriever Node performs similarity search using FAISS.

Input: User Query
Process: Query → Multimodal Embedding → FAISS Search → Top-K Results
Output: Retrieved multimodal documents


Retrieved documents may include:

- Text chunks
- Tables
- Images
- Page renderings

---

### Answer Node

The Answer Node synthesizes the final response using:

- Retrieved text context
- Retrieved image context
- System prompt grounding
- User query


### ⚡ Quick Start

##### Clone the Repository

git clone https://github.com/raulmanish73/Solution_Manish_Raul.git

cd Solution_Manish_Raul

##### Create Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate

##### Install Dependencies

pip install -r requirements.txt

##### Run Jupter notebook

final_solution_with_new_approach.ipynb
leadership_insight_agent.ipynb

