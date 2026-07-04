# Design Document

**Project:** AskMyBook — RAG-Powered Document Question Answering System  
**Author:** Shardul  
**Project Type:** Applied Machine Learning  
**Domain:** Natural Language Processing (NLP) & Large Language Models (LLMs)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [System Overview](#4-system-overview)
5. [Architecture](#5-architecture)
6. [Technology Stack](#6-technology-stack)
7. [Functional Requirements](#7-functional-requirements)
8. [Non-Functional Requirements](#8-non-functional-requirements)
9. [Workflow](#9-workflow)
10. [System Modules](#10-system-modules)
11. [Data Flow](#11-data-flow)
12. [Folder Structure](#12-folder-structure)
13. [Testing Strategy](#13-testing-strategy)
14. [Future Improvements](#14-future-improvements)
15. [Conclusion](#15-conclusion)

---

## 1. Introduction

**AskMyBook** is an AI-powered document question answering application that enables students to upload academic PDFs — textbooks, research papers, or college handbooks — and ask questions in plain English.

Instead of manually skimming through hundreds of pages, the system uses **Retrieval-Augmented Generation (RAG)** to find the most relevant sections of the uploaded document and generate accurate, cited answers using a Large Language Model (LLM).

Built as part of the StudyBuddy EdTech internship programme (I2 track), Summer 2026.

---

## 2. Problem Statement

Students regularly need to extract specific information from long academic PDFs. The current workflow — Ctrl+F, skim, repeat — is slow and fails entirely for semantic questions like *"what does the author say about regularisation?"*

Traditional keyword search cannot understand context or semantics, and provides no source traceability. This project builds an intelligent assistant that understands user queries, retrieves the relevant document sections, and generates grounded, citation-backed answers.

---

## 3. Objectives

### Primary Objectives
- Upload and process PDF documents
- Extract and clean document text with page-level metadata
- Generate semantic embeddings using an open-source model
- Store and index embeddings in a vector database
- Retrieve relevant chunks using hybrid (BM25 + dense) search
- Generate accurate, cited answers using an LLM
- Refuse or fall back gracefully when evidence is weak or the question is out-of-corpus

### Secondary Objectives
- Modular, readable codebase
- User-friendly Streamlit interface with inline citations
- Evaluation suite (20+ Q&A pairs) with transparent scoring
- Mini-extension: Compare Two Documents
- Future scalability to multi-source enterprise corpora

---

## 4. System Overview

The system consists of seven stages:

1. Document Upload
2. Text Extraction & Cleaning
3. Text Chunking
4. Embedding Generation
5. Hybrid Retrieval (BM25 + Dense)
6. Guardrail Check
7. Answer Generation with Citations

---

## 5. Architecture

```
                      User
                        │
                        ▼
               Streamlit Interface
                        │
                        ▼
               PDF Document Upload
                        │
                        ▼
               Text Extraction Layer
               (PyMuPDF — page-aware)
                        │
                        ▼
               Text Chunking Module
               (semantic + structural)
                        │
                        ▼
              Embedding Generation
              (BGE-small-en, open-source)
                        │
                        ▼
              ChromaDB Vector Database
                        │
           ┌────────────┴────────────┐
           ▼                         ▼
     User Question            Query Embedding
           └────────────┬────────────┘
                        ▼
             Hybrid Retrieval (BM25 + Dense)
             Reciprocal Rank Fusion (RRF)
                        │
                        ▼
               Guardrail Check
         (low-confidence → "I don't know"
          out-of-corpus → refusal)
                        │
                        ▼
          Prompt Construction Layer
                        │
                        ▼
          Large Language Model (LLM)
               (Claude Haiku)
                        │
                        ▼
         Generated Response + Citations
                  (page numbers)
```

---

## 6. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Frontend | Streamlit |
| PDF Parser | PyMuPDF |
| Chunking | Custom (semantic + structural) |
| Embeddings | BGE-small-en (open-source) |
| Sparse Retrieval | BM25 (rank-bm25) |
| Vector Database | ChromaDB |
| Retrieval Fusion | Reciprocal Rank Fusion (RRF) |
| LLM | Claude Haiku (Anthropic API) |
| Evaluation | Hand-rolled + LLM-as-judge |
| Version Control | Git & GitHub |

---

## 7. Functional Requirements

- Upload one or more PDF documents via the UI
- Extract text with page-level metadata preserved
- Split text into overlapping chunks (semantic + structural boundaries)
- Generate embeddings for every chunk using BGE-small-en
- Store vectors and metadata in ChromaDB
- Perform hybrid similarity search (BM25 + dense, fused via RRF)
- Apply guardrails: low-confidence fallback and out-of-corpus refusal
- Generate grounded responses with inline page citations
- Display source passages alongside each answer
- Mini-extension: retrieve from two documents and generate a comparative answer

---

## 8. Non-Functional Requirements

- Fast retrieval (sub-2s for typical academic PDF)
- Reliable PDF parsing across text-native documents
- Modular codebase (each stage independently testable)
- Scalable vector store (swap ChromaDB for Qdrant in production)
- Transparent evaluation with reproducible scores

---

## 9. Workflow

1. Upload PDF document(s) via the Streamlit interface.
2. Extract document text page-by-page using PyMuPDF.
3. Clean text: strip headers/footers, fix hyphenation, normalise whitespace.
4. Split text into chunks (≤400 tokens, ~50-token overlap, respecting paragraph boundaries).
5. Generate embeddings for every chunk using BGE-small-en.
6. Store embeddings and metadata (`page`, `section_title`, `chunk_id`) in ChromaDB.
7. Convert the user query into an embedding.
8. Run BM25 sparse retrieval → top-20 candidates.
9. Run dense vector search → top-20 candidates.
10. Fuse results using Reciprocal Rank Fusion → top-5 chunks.
11. Check confidence threshold; trigger guardrail if below threshold.
12. Build a prompt using the retrieved chunks with page references.
13. Generate the final answer using Claude Haiku.
14. Display answer + cited page numbers in the UI.

---

## 10. System Modules

| File | Responsibility |
|---|---|
| `ingest/pdf_loader.py` | PDF parsing, page extraction, metadata tagging |
| `ingest/text_splitter.py` | Chunking (semantic + structural, overlap) |
| `embed/embeddings.py` | Embedding generation (BGE-small-en) |
| `retrieval/vector_store.py` | ChromaDB setup, insert, query |
| `retrieval/bm25_retriever.py` | BM25 sparse retrieval |
| `retrieval/hybrid_retriever.py` | RRF fusion of BM25 + dense results |
| `generation/prompt_builder.py` | Prompt template with context + citation format |
| `generation/llm_client.py` | Claude Haiku API calls |
| `generation/guardrails.py` | Confidence threshold + out-of-corpus refusal |
| `eval/run_eval.py` | 20+ Q&A evaluation pipeline |
| `app.py` | Streamlit UI entry point |

---

## 11. Data Flow

```
PDF Upload
    ↓
Text Extraction (PyMuPDF, page-aware)
    ↓
Cleaning & Chunking
    ↓
Embedding Generation (BGE-small-en)
    ↓
ChromaDB (vectors + metadata)
    ↓
User Query
    ↓
Query Embedding
    ↓
BM25 + Dense Retrieval
    ↓
RRF Fusion → Top-5 Chunks
    ↓
Guardrail Check
    ↓
Prompt Construction
    ↓
Claude Haiku (LLM)
    ↓
Answer + Page Citations
```

---

## 12. Folder Structure

```
askmybook/
├── app.py                     # Streamlit UI entry point
├── README.md
├── design_doc.md
├── requirements.txt
├── .env.example
├── ingest/
│   ├── pdf_loader.py
│   └── text_splitter.py
├── embed/
│   └── embeddings.py
├── retrieval/
│   ├── vector_store.py
│   ├── bm25_retriever.py
│   └── hybrid_retriever.py
├── generation/
│   ├── prompt_builder.py
│   ├── llm_client.py
│   └── guardrails.py
├── eval/
│   ├── qa_pairs.json
│   └── run_eval.py
├── docs/
│   └── adrs/
│       ├── adr_001_pdf_parser.md
│       ├── adr_002_embeddings.md
│       └── adr_003_vector_db.md
├── data/                      # Uploaded PDFs (gitignored)
├── vectorstore/               # ChromaDB files (gitignored)
└── assets/                    # UI assets, screenshots
```

---

## 13. Testing Strategy

- **Unit testing** — each module (chunker, embedder, retriever) tested independently with sample inputs
- **Integration testing** — end-to-end pipeline tested on a small reference PDF with known answers
- **Retrieval quality testing** — precision@5 measured on the 20+ Q&A eval set
- **Guardrail testing** — out-of-corpus and low-confidence queries tested explicitly
- **User acceptance testing** — demo walkthrough with a peer who hasn't seen the project

---

## 14. Future Improvements

- OCR support for scanned PDFs (pytesseract)
- Conversation memory (multi-turn Q&A)
- Multi-document enterprise corpus (4+ source types)
- Fine-tuning a small model on the domain
- Agentic patterns for complex multi-hop queries
- Cloud deployment with authentication
- Support for non-PDF formats (Word, HTML, Markdown)

---

## 15. Conclusion

AskMyBook demonstrates a complete, production-minded Retrieval-Augmented Generation (RAG) pipeline — combining hybrid semantic search, a vector database, guardrails, and a Large Language Model to deliver accurate, cited answers from uploaded academic documents.

The modular architecture is designed for maintainability, clear evaluation, and a direct upgrade path to the 3rd-year enterprise RAG track. The mini-extension (Compare Two Documents) adds a multi-document reasoning layer that distinguishes this project from a basic chatbot implementation.
