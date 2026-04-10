#!/usr/bin/env python3
"""
MedHopQA RAG v2 (Zero Hardcoding)
- Sparse + Dense retrieval with Reciprocal Rank Fusion (RRF)
- Multi-candidate hop-2 bridge generation
- Cross-encoder reranking
- Sentence-level reranking
- Answer-type prediction
- Extraction-style final answering
- Semantic grading with safer judge parsing

Example:
python medhopqa_rag.py \
  --dev_csv dev.csv \
  --chunks wiki_chunks.jsonl \
  --save_preds rag_predictions.jsonl \
  --ollama_model qwen3:8b \
  --ollama_host http://localhost:11434
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm


# ---------------------------
# Data Loading & Utilities
# ---------------------------

def build_retrieval_text(rec: Dict) -> str:
    parts = []
    for key in ("title", "section", "subtitle", "header", "text"):
        val = rec.get(key, "")
        if val:
            val = str(val).strip()
            if val:
                parts.append(val)
    if not parts:
        parts.append(str(rec.get("text", "") or ""))
    return " | ".join(parts).strip()


def load_chunks(chunks_path: Path) -> Tuple[List[str], List[Dict]]:
    texts, meta = [], []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(build_retrieval_text(rec))
            meta.append(rec)
    return texts, meta


def find_short_answer_col(cols: List[str]) -> str:
    for c in cols:
        if "short" in c.strip().lower():
            return c
    raise ValueError("Could not find 'Short Answer' column.")


def basic_normalize(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(s).lower())).strip()


def clean_raw_llm_output(raw_text: str) -> str:
    text = str(raw_text).strip()

    text = re.sub(r"^ANSWER:\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\n.*$", "", text, flags=re.S).strip()

    for separator in [" and ", " or ", ";", ","]:
        if separator in text:
            text = text.split(separator)[0].strip()
            break

    text = re.sub(r"\(.*?\)", "", text).strip()
    text = re.sub(r"^(the answer is|answer|final answer)\s*:?\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\s+", " ", text).strip(" .;:'\"-")
    return text


def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def dedupe_keep_order(items: List[str]) -> List[str]:
    out, seen = [], set()
    for x in items:
        key = basic_normalize(x)
        if key and key not in seen:
            seen.add(key)
            out.append(x)
    return out


def rrf_fuse(rank_lists: List[List[int]], k: int = 60) -> List[int]:
    scores = {}
    for lst in rank_lists:
        for rank, idx in enumerate(lst, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return [idx for idx, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def safe_post_json(url: str, payload: Dict, timeout: int = 60) -> Dict:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------------------
# LLM Interactions (Ollama)
# ---------------------------

def predict_answer_type(question: str, model: str, host: str) -> str:
    prompt = (
        "Classify the expected answer type for the question.\n"
        "Output exactly one label from this list:\n"
        "YESNO, NUMBER, GENE, PROTEIN, DISEASE, SYNDROME, DRUG, PROCEDURE, CHROMOSOME, "
        "BONE, ORGAN, SPECIALIST, SYMPTOM, PHASE, PERSON, PLACE, PUBLISHER, CHEMICAL, OTHER\n\n"
        f"QUESTION: {question}\n\n"
        "LABEL:"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 5}
    }
    try:
        data = safe_post_json(f"{host.rstrip('/')}/api/generate", payload, timeout=30)
        label = data.get("response", "").strip().upper()
        label = re.sub(r"[^A-Z]", "", label)
        allowed = {
            "YESNO", "NUMBER", "GENE", "PROTEIN", "DISEASE", "SYNDROME", "DRUG",
            "PROCEDURE", "CHROMOSOME", "BONE", "ORGAN", "SPECIALIST", "SYMPTOM",
            "PHASE", "PERSON", "PLACE", "PUBLISHER", "CHEMICAL", "OTHER"
        }
        return label if label in allowed else "OTHER"
    except Exception:
        return "OTHER"


def generate_hop2_queries(question: str, initial_chunks: List[str], model: str, host: str) -> List[str]:
    evidence_text = "\n\n".join([f"- {chunk}" for chunk in initial_chunks[:5]])
    prompt = (
        "You are a strict biomedical retrieval assistant.\n"
        "From the QUESTION and EVIDENCE, extract up to 3 candidate missing-link entities "
        "that would be useful as follow-up search queries.\n"
        "Output one entity per line.\n"
        "Each line must be 1 to 5 words.\n"
        "No numbering. No explanations. No sentences.\n\n"
        f"QUESTION: {question}\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        "CANDIDATE ENTITIES:"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 30}
    }
    try:
        data = safe_post_json(f"{host.rstrip('/')}/api/generate", payload, timeout=60)
        raw = data.get("response", "").strip()
        lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", x).strip(" \"'") for x in raw.splitlines()]
        lines = [x for x in lines if x and len(x.split()) <= 6]
        return dedupe_keep_order(lines)[:3]
    except Exception:
        return []


def build_answer_constraints(answer_type: str) -> str:
    rules = [
        "- Output 1 to 5 words when possible.",
        "- Prefer exact wording from the evidence.",
        "- If unsupported, output exactly: Unknown."
    ]
    if answer_type == "YESNO":
        rules.append("- Output exactly Yes or No.")
    elif answer_type == "NUMBER":
        rules.append("- Output only the number or shortest number phrase supported by the evidence.")
    elif answer_type == "CHROMOSOME":
        rules.append("- Output a chromosome format answer such as Chromosome 11, if supported.")
        rules.append("- Do not output a gene or protein name.")
    elif answer_type == "GENE":
        rules.append("- Output only the gene symbol or gene name, not the disease.")
    elif answer_type == "PROTEIN":
        rules.append("- Output only the protein name, not the gene unless the evidence clearly uses the gene symbol as the answer.")
    elif answer_type == "BONE":
        rules.append("- Output only the anatomical bone name.")
    elif answer_type == "PHASE":
        rules.append("- Output only the cell-cycle phase name.")
    elif answer_type == "PUBLISHER":
        rules.append("- Output only the publisher or press name.")
    elif answer_type == "SPECIALIST":
        rules.append("- Output only the specialist title.")
    elif answer_type == "SYMPTOM":
        rules.append("- Output the symptom or sign, not the syndrome or disease.")
    elif answer_type == "PROCEDURE":
        rules.append("- Output the procedure name only.")
    elif answer_type == "DRUG":
        rules.append("- Output the drug name only.")
    elif answer_type == "PERSON":
        rules.append("- Output the person's name only.")
    elif answer_type == "PLACE":
        rules.append("- Output the place name only.")
    elif answer_type == "CHEMICAL":
        rules.append("- Output the chemical name only.")
    return "\n".join(rules)


def generate_llm_answer(question: str, evidence_passages: List[str], answer_type: str, model: str, host: str) -> str:
    evidence_text = "\n".join([f"[{i+1}] {p}" for i, p in enumerate(evidence_passages[:12])])
    prompt = (
        "You are a biomedical answer extractor.\n"
        "Select the single best answer supported by the evidence.\n"
        f"Expected answer type: {answer_type}\n\n"
        "Rules:\n"
        f"{build_answer_constraints(answer_type)}\n\n"
        f"QUESTION: {question}\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        "Output exactly in this format:\n"
        "ANSWER: <answer>\n"
        "EVIDENCE_ID: <number>\n"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "top_p": 0.9, "num_predict": 40}
    }
    try:
        data = safe_post_json(f"{host.rstrip('/')}/api/generate", payload, timeout=120)
        raw = data.get("response", "").strip()
        m = re.search(r"ANSWER:\s*(.+)", raw)
        ans = m.group(1).strip() if m else raw.strip()
        return clean_raw_llm_output(ans)
    except Exception:
        return "Unknown"


# ---------------------------
# Sentence Reranking
# ---------------------------

def rank_sentences(question: str, chunks: List[str], reranker, top_k: int = 12) -> List[str]:
    sentences = []
    for ch in chunks:
        sentences.extend(split_into_sentences(ch))
    sentences = dedupe_keep_order(sentences)
    if not sentences:
        return []

    pairs = [[question, s] for s in sentences]
    scores = reranker.predict(pairs, batch_size=64)
    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:top_k]]


# ---------------------------
# Grading
# ---------------------------

def semantic_match(question: str, prediction: str, gold: str, model: str, host: str) -> bool:
    norm_p = basic_normalize(prediction)
    norm_g = basic_normalize(gold)

    if norm_p == norm_g:
        return True

    if len(norm_p) > 3 and len(norm_g) > 3:
        if norm_p in norm_g or norm_g in norm_p:
            return True

    if not prediction or prediction.lower() == "unknown":
        return False

    prompt = (
        "You are an expert medical grader evaluating an AI system.\n"
        "Determine if the predicted answer is medically, semantically, or practically equivalent to the gold standard answer.\n\n"
        "Return exactly one word: True or False.\n\n"
        f"QUESTION: {question}\n"
        f"PREDICTED ANSWER: {prediction}\n"
        f"GOLD STANDARD: {gold}\n"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 5}
    }
    try:
        data = safe_post_json(f"{host.rstrip('/')}/api/generate", payload, timeout=60)
        response_text = data.get("response", "").strip().lower()
        first = response_text.split()[0] if response_text.split() else ""
        return first in {"true", "yes"}
    except Exception:
        return False


# ---------------------------
# Retrieval Helpers
# ---------------------------

def retrieve_tfidf(query: str, vectorizer, X_tfidf, top_n: int = 100) -> List[int]:
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X_tfidf).ravel()
    return sims.argsort()[::-1][:top_n].tolist()


def retrieve_dense(query: str, dense_model, X_dense, top_n: int = 100) -> List[int]:
    qv = dense_model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(qv, X_dense).ravel()
    return sims.argsort()[::-1][:top_n].tolist()


# ---------------------------
# Pipeline Execution
# ---------------------------

def run_pipeline(args):
    print("Loading chunks...")
    chunk_texts, chunk_meta = load_chunks(args.chunks)
    print(f"Loaded {len(chunk_texts)} chunks.")

    print("Building TF-IDF index...")
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=args.max_features)
    X_tfidf = tfidf_vectorizer.fit_transform(chunk_texts)

    print(f"Loading dense encoder: {args.dense_model}")
    dense_vectorizer = SentenceTransformer(args.dense_model)
    X_dense = dense_vectorizer.encode(
        chunk_texts,
        show_progress_bar=True,
        batch_size=args.dense_batch_size,
        normalize_embeddings=True
    )

    print(f"Loading Cross-Encoder: {args.cross_encoder_model}")
    reranker = CrossEncoder(args.cross_encoder_model)

    print("Loading dev CSV...")
    df = pd.read_csv(args.dev_csv)
    short_col = find_short_answer_col(list(df.columns))
    questions = df["Question"].astype(str).tolist()
    golds = df[short_col].astype(str).tolist()

    results = []
    correct = 0

    for q, gold in tqdm(list(zip(questions, golds)), total=len(questions)):
        answer_type = predict_answer_type(q, args.ollama_model, args.ollama_host)

        # Hop 1 retrieval
        top_idx_1_tfidf = retrieve_tfidf(q, tfidf_vectorizer, X_tfidf, top_n=args.first_hop_top_n)
        top_idx_1_dense = retrieve_dense(q, dense_vectorizer, X_dense, top_n=args.first_hop_top_n)
        hop1_combined_idx = rrf_fuse([top_idx_1_tfidf, top_idx_1_dense])[:args.hop_pool_size]

        hop1_chunks_for_bridge = [chunk_texts[i] for i in hop1_combined_idx[:5]]
        hop2_queries = generate_hop2_queries(q, hop1_chunks_for_bridge, args.ollama_model, args.ollama_host)

        # Hop 2 retrieval across multiple bridge candidates
        all_rank_lists = [hop1_combined_idx]
        combined_queries = [q] + [f"{q} {hq}".strip() for hq in hop2_queries if hq]

        for cq in combined_queries[1:]:
            idx_tfidf = retrieve_tfidf(cq, tfidf_vectorizer, X_tfidf, top_n=args.second_hop_top_n)
            idx_dense = retrieve_dense(cq, dense_vectorizer, X_dense, top_n=args.second_hop_top_n)
            all_rank_lists.append(rrf_fuse([idx_tfidf, idx_dense])[:args.hop_pool_size])

        all_candidate_idx = rrf_fuse(all_rank_lists)[:args.final_candidate_pool]

        # Chunk reranking
        pairs = [[q, chunk_texts[i]] for i in all_candidate_idx]
        scores = reranker.predict(pairs, batch_size=args.rerank_batch_size)
        ranked_pairs = sorted(zip(all_candidate_idx, scores), key=lambda x: x[1], reverse=True)

        final_top_chunks = [chunk_texts[i] for i, _ in ranked_pairs[:args.top_chunks_for_sentences]]
        final_top_sentences = rank_sentences(q, final_top_chunks, reranker, top_k=args.top_sentences)

        if not final_top_sentences:
            final_top_sentences = final_top_chunks[: min(8, len(final_top_chunks))]

        raw_prediction = generate_llm_answer(
            q,
            final_top_sentences,
            answer_type,
            args.ollama_model,
            args.ollama_host
        )
        prediction = clean_raw_llm_output(raw_prediction)

        # Retry on Unknown with tighter evidence
        if prediction.lower() == "unknown" and final_top_sentences:
            retry_evidence = final_top_sentences[:3]
            raw_prediction = generate_llm_answer(
                q,
                retry_evidence,
                answer_type,
                args.ollama_model,
                args.ollama_host
            )
            prediction = clean_raw_llm_output(raw_prediction)

        is_correct = semantic_match(q, prediction, gold, args.ollama_model, args.ollama_host)
        if is_correct:
            correct += 1

        results.append({
            "question": q,
            "answer_type": answer_type,
            "hop2_queries_used": hop2_queries,
            "prediction": prediction,
            "gold": gold,
            "is_correct": is_correct
        })

    acc = correct / len(questions) if questions else 0.0
    print(f"\nFinal Accuracy (Semantic Match): {correct}/{len(questions)} = {acc:.3f}")

    with args.save_preds.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved predictions -> {args.save_preds}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_csv", type=Path, required=True, help="Batch eval CSV")
    ap.add_argument("--chunks", type=Path, required=True, help="JSONL chunk file")
    ap.add_argument("--save_preds", type=Path, default=Path("rag_predictions.jsonl"))

    ap.add_argument("--ollama_model", type=str, default="qwen3:8b")
    ap.add_argument("--ollama_host", type=str, default="http://localhost:11434")

    ap.add_argument("--dense_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--cross_encoder_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-12-v2")

    ap.add_argument("--max_features", type=int, default=250000)
    ap.add_argument("--dense_batch_size", type=int, default=64)
    ap.add_argument("--rerank_batch_size", type=int, default=32)

    ap.add_argument("--first_hop_top_n", type=int, default=100)
    ap.add_argument("--second_hop_top_n", type=int, default=100)
    ap.add_argument("--hop_pool_size", type=int, default=150)
    ap.add_argument("--final_candidate_pool", type=int, default=300)
    ap.add_argument("--top_chunks_for_sentences", type=int, default=25)
    ap.add_argument("--top_sentences", type=int, default=12)

    args = ap.parse_args()
    run_pipeline(args)
