# MedhopQA RAG - Biomedical Multi-Hop Question Answering

A modular **Retrieval-Augmented Generation (RAG)** system for biomedical multi-hop question answering, built on the MedHop dataset. The system answers complex questions requiring multi-step reasoning across multiple documents involving diseases, genes, and chemical entities.

## Overview

Standard QA systems struggle with questions that cannot be answered from a single passage — they require chaining evidence across multiple sources. This project implements a full RAG pipeline with hybrid retrieval, grounded answer generation, and a systematic evaluation framework to tackle this challenge in the biomedical domain.

## Tech Stack

- **Language:** Python 3.10+
- **Sparse Retrieval:** BM25 (rank_bm25), TF-IDF
- **Dense Retrieval:** FAISS, BioBERT embeddings (HuggingFace Transformers)
- **Hybrid Retrieval:** Weighted score fusion of sparse + dense results
- **LLM Generation:** OpenAI API (GPT-4)
- **Evaluation:** Exact Match, F1 Score, answer normalization
- **Data:** MedHop / Biochem biomedical QA dataset

## Architecture

The pipeline takes a natural language query, runs it through both BM25 sparse retrieval and BioBERT dense vector search (FAISS), fuses the ranked results, builds an evidence-conditioned prompt, and passes it to GPT-4 for grounded short-answer generation.

## Key Features

- **Hybrid Retrieval** - Combines BM25 sparse search with BioBERT dense vector search for improved recall over either method alone
- **Multi-Hop Reasoning** - Handles questions requiring evidence chaining across diseases, genes, and chemical entities
- **Grounded Generation** - LLM answers strictly conditioned on retrieved context with fallback behavior when evidence is insufficient, reducing hallucinations
- **Evaluation Pipeline** - Exact Match and F1 scoring with answer normalization and failure categorization across retrieval, reasoning, and generation error types
- **Prompt Engineering** - Experiments with query rewriting, chain-of-thought prompting, and missing-link entity extraction for complex multi-step questions

## Project Structure

```
MedhopQA_RAG/
├── medhopqa_rag.py      # Main RAG pipeline (retrieval + generation + evaluation)
├── Biochemdata.zip      # MedHop-style biomedical QA dataset
└── README.md
```

## Setup & Usage

```bash
# Install dependencies
pip install openai faiss-cpu rank_bm25 transformers sentence-transformers

# Set your OpenAI API key
export OPENAI_API_KEY=your_key_here

# Run the pipeline
python medhopqa_rag.py
```

## Skills Demonstrated

- Retrieval-Augmented Generation (RAG) pipeline design and implementation
- Hybrid sparse + dense retrieval with FAISS and BM25
- Biomedical NLP using HuggingFace Transformers (BioBERT)
- LLM prompt engineering for grounded, hallucination-resistant generation
- End-to-end evaluation with Exact Match / F1 scoring and failure analysis
- Multi-document reasoning over biomedical knowledge graphs

---

*Research project - Western University, 2026.*
