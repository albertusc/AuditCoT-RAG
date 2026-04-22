# RAG-Audit: Financial Document Intelligence System

A comprehensive RAG (Retrieval-Augmented Generation) pipeline for Indonesian financial documents,
featuring document processing, chunking strategies, hybrid retrieval, and fine-tuned LLM evaluation.


## Overview

This project implements an end-to-end system for extracting, processing, and querying financial
information from Indonesian annual reports and financial statements. The system:

- Extracts text from 120+ PDF documents (annual reports + financial statements) across 15 Indonesian
  public companies (BBCA, BBRI, BMRI, TLKM, ASII, etc.)
- Implements multiple chunking strategies optimized for financial tables and narrative text
- Builds a hybrid retrieval system combining FAISS dense embeddings (multilingual-e5-base) and BM25
  sparse retrieval with cross-encoder reranking
- Fine-tunes Qwen3-8B using QLoRA for financial QA
- Evaluates performance across multiple metrics (Consistency Score, Matched Answer, Answer Reasoning Quality)


## System Architecture

```
+-----------------------------------------------------------------------------+
|                          DATA EXTRACTION PIPELINE                           |
+-----------------------------------------------------------------------------+
|  PDF Documents -> Marker (OCR/Layout) -> JSONL -> Text Cleaning -> Enrichment  |
|     120 files         (multilingual)        per page    +metadata           |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                            CHUNKING STRATEGIES                              |
+-----------------------------------------------------------------------------+
|  - Element-based splitting (headers, tables, paragraphs)                    |
|  - Financial section classification (balance sheet, income statement, etc.) |
|  - Table preservation with caption attachment                               |
|  - 2,048 character target chunk size                                        |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                         HYBRID RETRIEVAL SYSTEM                             |
+-----------------------------------------------------------------------------+
|                                                                             |
|   Query --+-> BM25 (Sparse) --+                                             |
|           |                   +--> RRF Fusion --> Cross-Encoder --> Top-K   |
|           +-> FAISS (Dense) --+      (k=600)       (BGE-reranker)           |
|                                                                             |
|   Embedding Model : intfloat/multilingual-e5-base (768-dim)                 |
|   Cross-Encoder   : BAAI/bge-reranker-v2-m3                                 |
|   Vector DB       : FAISS IndexFlatIP (cosine similarity via L2 norm)       |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                          QGA DATASET GENERATION                             |
+-----------------------------------------------------------------------------+
|  - Gemini-2.5-flash generated Question-Guidance-Answer pairs                |
|  - Temporal split: Train (2021-2022) / Test (2023-2024)                     |
|  - Strict confidence threshold: >= 0.95                                     |
|  - Cognitive skills: direct retrieval, arithmetic, comparative              |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                           FINE-TUNING (QLoRA)                               |
+-----------------------------------------------------------------------------+
|   Base Model       : Qwen3-8B (4-bit quantized)                             |
|   LoRA Rank        : 8 | Alpha: 16 | Dropout: 0.10                          |
|   Trainable Params : 21.8M (0.27% of total)                                 |
|   Epochs           : 3 | Batch Size: 1 (gradient accumulation: 4)           |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                              EVALUATION                                     |
+-----------------------------------------------------------------------------+
|   Metrics    : CS (Consistency) | MA (Accuracy) | ARQ (Format Quality)      |
|   Comparison : BASELINE vs FINE-TUNED                                       |
+-----------------------------------------------------------------------------+
```


## Dataset Statistics

| Sector                  | Documents | Total Pages | Avg Pages |
|-------------------------|-----------|-------------|-----------|
| Keuangan (Finance)      | 24        | 14,139      | 589.1     |
| Energi (Energy)         | 24        | 6,288       | 262.0     |
| Consumer                | 24        | 5,836       | 243.2     |
| Basic Materials         | 24        | 5,111       | 213.0     |
| Infrastruktur           | 8         | 2,274       | 284.2     |
| Kesehatan (Healthcare)  | 8         | 1,969       | 246.1     |
| Properti (Property)     | 8         | 1,730       | 216.2     |
| Total                   | 120       | 23,618      | 196.8     |


## Key Results

### Performance Comparison

| Metric                          | Baseline | Fine-Tuned | Improvement |
|---------------------------------|----------|------------|-------------|
| Consistency Score (CS)          | 0.611    | 0.911      | +49.1%      |
| Matched Answer (MA)             | 0.178    | 0.300      | +68.8%      |
| Answer Reasoning Quality (ARQ)  | 0.511    | 0.994      | +94.6%      |

### Efficiency Metrics

| Metric                  | Baseline | Fine-Tuned | Change  |
|-------------------------|----------|------------|---------|
| Average Inference Time  | 58.16s   | 22.60s     | -61.2%  |
| Peak VRAM Usage         | 7.9 GB   | 8.0 GB     | +1.0%   |
| Average Output Tokens   | 489      | 90         | -81.5%  |


## Project Structure

```
RAG-Audit/
├── data/
│   ├── raw/
│   │   ├── Annual-Report/            # 60 annual reports (2021-2024)
│   │   └── Financial-Report/         # 60 financial statements (2021-2024)
│   ├── processed/
│   │   ├── extracted_pdfs_marker/    # Marker JSONL output
│   │   ├── enriched_pdfs/            # Chunks with metadata
│   │   ├── chunked_pdfs/             # Final 2,048-char chunks
│   │   └── qga_final_dataset/        # Train/test QGA pairs
├── Models/
│   ├── intfloat-multilingual-e5-base/
│   ├── bge-reranker-v2-m3/
│   ├── Qwen3-8B-4Bit-Quantized/
│   └── Qwen3-8B-QLoRA-Finetuned/
├── RAG Audit-CoT (Cleaned).ipynb
└── thesis_evaluation_final.json
```


## Technical Implementation Details

### Chunking Strategies

Element-Based Splitting (Logical Chunking):
- Preserves table boundaries using markdown pipe detection
- Forces new chunks on headers (# markdown syntax)
- Merges narrative text until 2,048 character limit
- Financial section classification via bilingual keyword detection

Table Preservation:

```python
# Tables are detected and preserved as atomic units
is_table = '|' in block and '---' in block and len(block.split('\n')) > 2
if is_table:
    # Flush current buffer, preserve entire table
    chunks.append(table_chunk)
```

### Hybrid Retrieval

```python
# Reciprocal Rank Fusion (RRF)
candidates = {}
for chunk_idx in top_bm25:
    candidates[chunk_idx] = candidates.get(chunk_idx, 0) + (bm25_scores[chunk_idx] * 0.5)
for i, chunk_idx in enumerate(faiss_top):
    candidates[chunk_idx] = candidates.get(chunk_idx, 0) + (faiss_scores[i] * 0.5)

# Cross-encoder reranking on top-50 candidates
scores = reranker.predict([(query, chunk_text) for chunk in top_candidates])
```

### QLoRA Fine-Tuning Configuration

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.10,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM
)
```


## Installation

```bash
conda create -n rag_env python=3.10
conda activate rag_env

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate peft bitsandbytes
pip install sentence-transformers faiss-gpu rank-bm25
pip install marker-pdf google-generativeai
pip install pandas numpy matplotlib seaborn tqdm
```


## Usage

### 1. Data Extraction

```python
extracted_path = extract_all_pdfs_marker(
    pdf_info_list=pdf_files_with_info,
    resume=True  # Skip already processed files
)
```

### 2. Chunking and Enrichment

```python
enricher = TickerMetadataBuilder()
data, stats = enricher.enrich_single_file(jsonl_path)
```

### 3. Build Vector Index

```python
create_embeddings_and_index(texts, metadata, tokenizer, model)
create_bm25_from_faiss_metadata(FAISS_METADATA_PATH, BM25_INDEX_PATH)
```

### 4. Fine-Tuning

```python
trainer.train()
model.save_pretrained(OUTPUT_DIR)
```

### 5. Evaluation

```python
base_metrics = evaluate_loop(base_model, tokenizer, data, contexts, "BASELINE")
ft_metrics = evaluate_loop(ft_model, tokenizer, data, contexts, "FINE-TUNED")
```


## Evaluation Metrics

| Metric                          | Description                                              |
|---------------------------------|----------------------------------------------------------|
| CS (Consistency Score)          | Stability across two inference runs (1.0 if identical)   |
| MA (Matched Answer)             | Factual accuracy judged by Gemini-2.5-flash              |
| ARQ (Answer Reasoning Quality)  | JSON format compliance + reasoning field presence        |


## Error Analysis

Root causes of failed extractions (MA = 0.0):

| Error Category                | Count | Percentage |
|-------------------------------|-------|------------|
| Reasoning / Extraction Error  | 42    | 66.7%      |
| Format / Unit Ambiguity       | 13    | 20.6%      |
| Retrieval Missing (OCR/Index) | 5     | 7.9%       |
| Hallucination / Degeneration  | 3     | 4.8%       |


## Key Contributions

1. Domain-Specific Chunking: Element-based splitting optimized for financial tables with bilingual
   keyword detection
2. Hybrid Retrieval: RRF fusion of BM25 + FAISS with cross-encoder reranking
3. Efficient Fine-Tuning: QLoRA enables 8B parameter model training on consumer GPU
4. Comprehensive Evaluation: Multi-metric framework covering consistency, accuracy, and format quality
5. Indonesian Financial Corpus: 120 documents across 15 companies, 23,618 pages


## Citation

```bibtex
@misc{rag-audit-2026,
  title     = {AUDITRAG: OPTIMASI HYBRID RETRIEVAL DAN AUDIT CHAIN-OF-THOUGHT PADA MODEL QWEN-3 UNTUK PENALARAN NUMERIK LAPORAN KEUANGAN},
  author    = {Albertus Christian Wahyu Atmaja},
  year      = {2026},
  note      = {Undergraduate Thesis},
  publisher = {Universitas Multimedia Nusantara}
}
```


## License

This project is part of an undergraduate thesis. Contact the author for usage permissions.
