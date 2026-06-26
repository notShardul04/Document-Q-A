# AskMyBook

**AskMyBook** is a Retrieval-Augmented Generation (RAG) application that enables users to interact with PDF documents using natural language. Instead of relying solely on a large language model's internal knowledge, the system retrieves relevant information from uploaded documents and generates evidence-backed responses with citations.

This project is being developed as part of an Applied Machine Learning internship to demonstrate the implementation of a production-style RAG pipeline.

---

## Project Objectives

The primary objectives of this project are:

* Build an end-to-end Retrieval-Augmented Generation (RAG) system.
* Allow users to upload PDF documents and ask questions about their contents.
* Provide answers supported by document citations.
* Implement retrieval techniques that improve response accuracy.
* Evaluate the quality of generated answers using predefined metrics.
* Deploy the application for public access.

---

## Planned Features

### Document Ingestion

* Upload one or more PDF documents.
* Extract text from PDFs.
* Preserve document metadata such as page numbers.

### Document Processing

* Structural and semantic document chunking.
* Generation of vector embeddings.
* Storage of embeddings in a vector database.

### Retrieval Pipeline

* Dense vector search.
* BM25 keyword search.
* Hybrid retrieval for improved relevance.
* Top-k document retrieval.

### Question Answering

* Natural language interface.
* Context-aware answer generation.
* Evidence-based responses using retrieved document chunks.
* Inline citations containing page numbers and supporting text.

### Evaluation

* Evaluation dataset containing at least 20 test questions.
* Manual assessment of:

  * Correctness
  * Citation Precision
  * Completeness

### Guardrails

* Respond with "I don't know" when sufficient evidence is unavailable.
* Refuse questions unrelated to the uploaded documents.
* Prevent unsupported or hallucinated citations.

---

## Technology Stack

| Component            | Planned Technology             |
| -------------------- | ------------------------------ |
| Programming Language | Python 3.11                    |
| User Interface       | Streamlit                      |
| PDF Parsing          | PyMuPDF                        |
| Embedding Model      | BGE / OpenAI Embeddings        |
| Vector Database      | ChromaDB                       |
| Retrieval            | Hybrid Search (Dense + BM25)   |
| Language Model       | GPT-4o-mini or Open Source LLM |
| Evaluation           | RAGAS / Manual Evaluation      |
| Version Control      | Git and GitHub                 |

---

## Planned System Architecture

```text
PDF Documents
      │
      ▼
Document Parsing
      │
      ▼
Chunking
      │
      ▼
Embedding Generation
      │
      ▼
Vector Database
      │
      ▼
Hybrid Retrieval
      │
      ▼
Language Model
      │
      ▼
Response with Citations
```

---

## Project Structure

```text
AskMyBook/
│
├── app/
├── data/
│   ├── raw/
│   └── processed/
│
├── ingest/
├── embeddings/
├── retrieval/
├── prompts/
├── evaluation/
├── tests/
├── docs/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Development Roadmap

### Week 1

* Repository setup
* Development environment configuration
* PDF parsing
* Initial data ingestion

### Week 2

* Embedding generation
* Vector database integration
* Retrieval pipeline
* End-to-end prototype

### Week 3

* User interface improvements
* Testing
* Evaluation pipeline
* Documentation

### Week 4

* Deployment
* Performance optimization
* Guardrails
* Final documentation

---

## Planned Extension

### Compare Two Documents

The system will support comparative question answering across two uploaded documents. Users will be able to ask questions such as:

> "What are the differences between these two research papers regarding transformer architectures?"

The application will retrieve relevant information from both documents before generating a comparative response supported by citations.

---

## Learning Outcomes

This project aims to provide practical experience with:

* Retrieval-Augmented Generation (RAG)
* Document parsing and preprocessing
* Embedding models
* Vector databases
* Hybrid information retrieval
* Prompt engineering
* LLM evaluation
* AI guardrails
* End-to-end AI application development

---

## Project Status

Current Stage: Project Initialization

* [x] Repository created
* [x] Initial documentation
* [ ] PDF ingestion pipeline
* [ ] Document chunking
* [ ] Embedding generation
* [ ] Vector database integration
* [ ] Retrieval pipeline
* [ ] Question answering
* [ ] Evaluation framework
* [ ] Deployment

---

## License

This project is developed for educational and portfolio purposes.
