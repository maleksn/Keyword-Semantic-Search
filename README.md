# 🎬 Multimodal RAG & Hybrid Search Engine

A high-performance, end-to-end **Retrieval-Augmented Generation (RAG)** and **Information Retrieval (IR)** system built on a dataset of over 5,000 movies.

This repository explores and implements search architecture from scratch—transitioning from classical **Lexical (BM25 and TF-IDF)** indexing to **Dense Semantic Vector Search**, **Multimodal CLIP Embeddings**, **Hybrid Search with Reciprocal Rank Fusion (RRF)**, **LLM Re-ranking and Query Enhancement**, and production-grade **RAG Generation**.

---

## 📑 Table of Contents
- [Architecture Overview](#-architecture-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Quick Start Guide](#-quick-start-guide)
- [CLI Reference & Detailed Usage](#-cli-reference--detailed-usage)
  - [1. Keyword & BM25 Search](#1-keyword--bm25-search-keyword_search_clipy)
  - [2. Semantic Vector Search & Chunking](#2-semantic-vector-search--chunking-semantic_search_clipy)
  - [3. Hybrid Search & Advanced Re-ranking](#3-hybrid-search--advanced-re-ranking-hybrid_search_clipy)
  - [4. Multimodal & Vision Search](#4-multimodal--vision-search-multimodal_search_clipy--describe_image_clipy)
  - [5. Retrieval-Augmented Generation (RAG)](#5-retrieval-augmented-generation-rag-augmented_generation_clipy)
  - [6. Evaluation & IR Metrics](#6-evaluation--ir-metrics-evaluation_clipy)
- [Technical Deep Dive](#-technical-deep-dive)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🏛 Architecture Overview

```
                                  [ User Query / Image ]
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               ▼                                                         ▼
     [ Lexical Pipeline ]                                     [ Semantic Pipeline ]
  • Text Normalization & Stopwords                         • SentenceTransformers (all-MiniLM-L6-v2)
  • Porter Stemming (NLTK)                                 • Sentence-aware Semantic Chunking
  • Custom Inverted Index                                  • Dense Vector Embeddings (384-d)
  • BM25 Scoring (k1=1.5, b=0.75)                          • Cosine Similarity Scoring
               │                                                         │
               └────────────────────────────┬────────────────────────────┘
                                            │
                                  [ Hybrid Fusion ]
                       • Score Normalization (Min-Max)
                       • Weighted Linear Combination (Alpha)
                       • Reciprocal Rank Fusion (RRF, k=60)
                                            │
                                            ▼
                           [ Query Enhancement & Re-ranking ]
                       • Query Rewriting, Expansion, Spellcheck (LLM)
                       • Cross-Encoder (ms-marco-TinyBERT-L2-v2)
                       • Batch & Individual LLM Re-ranking
                                            │
                                            ▼
                               [ Augmented Generation ]
                       • Context Injection into Prompt
                       • OpenRouter / LLM Generation
                       • Fact Synthesizing & Bracketed Citations ([1], [2])
```

---

## 🚀 Key Features

* **Custom Inverted Index & BM25**: Built from the ground up without external search engines. Implements tokenization, stopword filtering, Porter stemming, Term Frequency (TF), Inverse Document Frequency (IDF), and tunable BM25 scoring ($k_1=1.5, b=0.75$).
* **Semantic Dense Retrieval**: Leverages `sentence-transformers/all-MiniLM-L6-v2` embeddings with cosine similarity matching.
* **Advanced Chunking Strategies**: Word-level sliding window chunking and sentence-aware boundary chunking with overlap to preserve narrative coherence.
* **Hybrid Search (Fusion)**:
  * **Score-based Weighted Search**: Blends normalized lexical and dense semantic scores using a tunable $\alpha$ parameter.
  * **Rank-based Fusion (RRF)**: Robust Reciprocal Rank Fusion ($k=60$) immune to score calibration discrepancies.
* **Query Transformation & Intelligence**:
  * Automated spellcheck correction via LLM.
  * Query rewriting tailored to streaming catalogs.
  * Domain-specific query expansion with relevant synonyms and keywords.
* **Two-Stage Re-ranking**:
  * **Cross-Encoder**: Local neural re-ranking using `cross-encoder/ms-marco-TinyBERT-L2-v2`.
  * **LLM Re-ranking**: Zero-shot individual scoring (0–10 scale) or batch list-wise ranking.
* **Multimodal Search**: Search movies using poster images via OpenAI CLIP (`clip-ViT-B-32`) joint embedding space, plus multimodal query rewriting using vision-language models.
* **Generation & Citations**: Multi-document summarization, conversational Q&A, and citation-backed factual grounding (`[1]`, `[2]`).
* **Quantitative Evaluation**: Precision@K, Recall@K, and F1-Score benchmarking against a curated golden test set (`data/golden_dataset.json`).

---

## 📂 Project Structure

```
.
├── cli/
│   ├── inverted_index.py           # Core Inverted Index, BM25, and TF-IDF implementation
│   ├── keyword_search_cli.py       # CLI interface for keyword and BM25 operations
│   ├── semantic_search_cli.py      # CLI for vector embeddings, chunking, and semantic search
│   ├── hybrid_search.py            # Hybrid fusion logic (Weighted & Reciprocal Rank Fusion)
│   ├── hybrid_search_cli.py        # CLI for hybrid search, query enhancement, and re-ranking
│   ├── describe_image_cli.py       # Vision LLM CLI for multimodal query reformulation
│   ├── multimodal_search_cli.py    # CLIP image-to-text vector search CLI
│   ├── augmented_generation_cli.py # RAG CLI (Q&A, summarization, citations)
│   ├── evaluation_cli.py           # Benchmark evaluation CLI (Precision@K, Recall@K, F1)
│   ├── test_llm.py                 # LLM connectivity check script
│   └── lib/
│       ├── search_utils.py         # Data loaders and formatting utilities
│       ├── semantic_search.py      # Semantic & Chunked search engine classes
│       └── multimodal_search.py    # CLIP search implementation
├── data/
│   ├── movies.json                 # Movie database (titles, descriptions, metadata)
│   ├── golden_dataset.json         # Evaluation test queries and ground-truth relevant docs
│   ├── stopwords.txt               # Custom stopword list for text filtering
│   └── paddington.jpeg             # Sample image for multimodal search testing
├── cache/                          # Serialized indices, embeddings, and metadata
├── pyproject.toml                  # Python project metadata and dependencies
├── uv.lock                         # Lockfile for reproducible dependencies
└── README.md                       # Documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
* **Python 3.11+**
* [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip` / `venv`
* An **OpenRouter API Key** (for LLM-powered query enhancement, re-ranking, and RAG generation)

### 1. Clone the Repository
```bash
git clone https://github.com/maleksn/Keyword-Semantic-Search.git
cd Keyword-Semantic-Search
```

### 2. Set Up Virtual Environment & Dependencies

Using **uv**:
```bash
# Create venv and install dependencies
uv sync
source .venv/bin/activate
```

Or using standard **venv** and **pip**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

---

## ⚡ Quick Start Guide

### Step 1: Build the Search Indices
Build the inverted index and precompute chunked embeddings for fast lookups:
```bash
# 1. Build lexical inverted index and document maps
python3 cli/keyword_search_cli.py build

# 2. Build dense chunked embeddings (downloads all-MiniLM-L6-v2 model if needed)
python3 cli/semantic_search_cli.py embed_chunks
```

### Step 2: Run a Hybrid Search
```bash
python3 cli/hybrid_search_cli.py rrf-search "time travel paradox" --limit 3
```

### Step 3: Ask a RAG Question
```bash
python3 cli/augmented_generation_cli.py citations "Which movies involve dream heists or mind manipulation?"
```

---

## 💻 CLI Reference & Detailed Usage

### 1. Keyword & BM25 Search (`keyword_search_cli.py`)

Provides direct access to the inverted index, statistical weights, and BM25 ranking.

* **Build the index**:
  ```bash
  python3 cli/keyword_search_cli.py build
  ```
* **Standard inverted index lookup**:
  ```bash
  python3 cli/keyword_search_cli.py search "batman"
  ```
* **Inspect Term Frequency (TF)**:
  ```bash
  python3 cli/keyword_search_cli.py tf <doc_id> <term>
  # Example:
  python3 cli/keyword_search_cli.py tf 929 arkham
  ```
* **Inspect Inverse Document Frequency (IDF)**:
  ```bash
  python3 cli/keyword_search_cli.py idf "superhero"
  ```
* **Inspect TF-IDF score**:
  ```bash
  python3 cli/keyword_search_cli.py tfidf 929 arkham
  ```
* **Inspect BM25 TF & BM25 IDF values**:
  ```bash
  python3 cli/keyword_search_cli.py bm25tf 929 arkham
  python3 cli/keyword_search_cli.py bm25idf arkham
  ```
* **Full BM25 search**:
  ```bash
  python3 cli/keyword_search_cli.py bm25search "dark knight gotham" --limit 5
  ```

---

### 2. Semantic Vector Search & Chunking (`semantic_search_cli.py`)

Utilizes `all-MiniLM-L6-v2` embeddings for dense semantic matching and chunk-level retrieval.

* **Verify sentence-transformer model loading**:
  ```bash
  python3 cli/semantic_search_cli.py verify
  ```
* **Generate & inspect embedding for raw text**:
  ```bash
  python3 cli/semantic_search_cli.py embed "A detective investigating a crime scene."
  ```
* **Verify precomputed document embeddings**:
  ```bash
  python3 cli/semantic_search_cli.py verify_embeddings
  ```
* **Full document semantic search**:
  ```bash
  python3 cli/semantic_search_cli.py search "space exploration and black holes" --limit 5
  ```
* **Test word-based chunking**:
  ```bash
  python3 cli/semantic_search_cli.py chunk "Long description text..." --chunk-size 50 --overlap 10
  ```
* **Test sentence-based semantic chunking**:
  ```bash
  python3 cli/semantic_search_cli.py semantic_chunk "Sentence one. Sentence two. Sentence three." --max-chunk-size 2 --overlap 1
  ```
* **Precompute chunked embeddings**:
  ```bash
  python3 cli/semantic_search_cli.py embed_chunks
  ```
* **Search over chunked embeddings**:
  ```bash
  python3 cli/semantic_search_cli.py search_chunked "heartwarming family animated adventure" --limit 5
  ```

---

### 3. Hybrid Search & Advanced Re-ranking (`hybrid_search_cli.py`)

Merges lexical BM25 and semantic search results using score-based or rank-based fusion, with optional query enhancement and secondary re-ranking.

#### Weighted Hybrid Search
Combines normalized BM25 and semantic scores using parameter `--alpha` ($\alpha \cdot \text{BM25} + (1-\alpha) \cdot \text{Semantic}$):
```bash
# Alpha = 0.7 (70% BM25, 30% Semantic)
python3 cli/hybrid_search_cli.py weighted-search "artificial intelligence uprising" --alpha 0.7 --limit 5
```

#### Reciprocal Rank Fusion (RRF)
Combines document rankings using RRF constant $k$ (default: 60):
```bash
python3 cli/hybrid_search_cli.py rrf-search "artificial intelligence uprising" -k 60 --limit 5
```

#### Query Enhancement via LLM
Enhance ambiguous or mistyped queries before retrieval using `--enhance`:
* `spell`: Fix typos while preserving query structure.
  ```bash
  python3 cli/hybrid_search_cli.py rrf-search "inceptoin dircted by nolan" --enhance spell
  ```
* `rewrite`: Convert conversational descriptions into focused search terms.
  ```bash
  python3 cli/hybrid_search_cli.py rrf-search "that bear movie where leo gets attacked" --enhance rewrite
  ```
* `expand`: Append synonyms and contextual keywords.
  ```bash
  python3 cli/hybrid_search_cli.py rrf-search "scary bear movie" --enhance expand
  ```

#### Two-Stage Re-ranking
Refine the initial candidate pool using second-stage scoring with `--rerank-method`:
* `cross_encoder`: Local cross-encoder model (`ms-marco-TinyBERT-L2-v2`).
  ```bash
  python3 cli/hybrid_search_cli.py rrf-search "cyberpunk dystopia" --rerank-method cross_encoder --limit 5
  ```
* `individual`: Score each candidate individually with an LLM prompt (0–10 scale).
  ```bash
  python3 cli/hybrid_search_cli.py rrf-search "cyberpunk dystopia" --rerank-method individual --limit 5
  ```
* `batch`: List-wise ranking of all candidates in a single LLM call.
  ```bash
  python3 cli/hybrid_search_cli.py rrf-search "cyberpunk dystopia" --rerank-method batch --limit 5
  ```

#### Inline LLM Evaluation
Evaluate retrieval quality on a 0–3 relevance scale using `--evaluate`:
```bash
python3 cli/hybrid_search_cli.py rrf-search "dramatic courtroom thriller" --evaluate
```

---

### 4. Multimodal & Vision Search (`multimodal_search_cli.py` & `describe_image_cli.py`)

Search movie descriptions using image queries via CLIP embeddings or formulate queries with vision models.

* **Verify CLIP image embeddings**:
  ```bash
  python3 cli/multimodal_search_cli.py verify_image_embedding data/paddington.jpeg
  ```
* **Search movies using an image query**:
  ```bash
  python3 cli/multimodal_search_cli.py image_search data/paddington.jpeg
  ```
* **Multimodal query rewriting using Vision LLM**:
  Synthesize text and image into an enhanced search query:
  ```bash
  python3 cli/describe_image_cli.py --image data/paddington.jpeg --query "What kind of movie is this?"
  ```

---

### 5. Retrieval-Augmented Generation (RAG) (`augmented_generation_cli.py`)

Combines hybrid retrieval with LLM generation for streaming catalog interactions.

* **Standard RAG answer**:
  ```bash
  python3 cli/augmented_generation_cli.py rag "Recommend three thriller movies with major plot twists"
  ```
* **Multi-document synthesis & summarization**:
  ```bash
  python3 cli/augmented_generation_cli.py summarize "What mafia movies are available?" --limit 5
  ```
* **Answers with grounded source citations**:
  Provides explicit bracketed citations (`[1]`, `[2]`) tied to retrieved documents:
  ```bash
  python3 cli/augmented_generation_cli.py citations "Which films feature Christopher Nolan as director?" --limit 5
  ```
* **Conversational persona Q&A**:
  Provides natural, conversational responses for interactive chat assistants:
  ```bash
  python3 cli/augmented_generation_cli.py question "Is there any good animated movie for kids tonight?"
  ```

---

### 6. Evaluation & IR Metrics (`evaluation_cli.py`)

Evaluates search system performance against `data/golden_dataset.json`:

* **Precision@K**: Fraction of retrieved movies that are relevant:
  $$\text{Precision@K} = \frac{|\text{Retrieved Relevant}|}{K}$$
* **Recall@K**: Fraction of all relevant movies that were retrieved:
  $$\text{Recall@K} = \frac{|\text{Retrieved Relevant}|}{|\text{All Relevant}|}$$
* **F1-Score**: Harmonic mean of Precision and Recall:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Run evaluation:
```bash
python3 cli/evaluation_cli.py --limit 5
```

---

## 🔬 Technical Deep Dive

### 1. BM25 Scoring
The Okapi BM25 implementation calculates:
$$\text{Score}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{\text{TF}(t, D) \cdot (k_1 + 1)}{\text{TF}(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
* $k_1 = 1.5$: Calibrates term frequency saturation.
* $b = 0.75$: Penalizes long documents relative to average document length.

### 2. Reciprocal Rank Fusion (RRF)
Combines heterogeneous rankings without score scale calibration:
$$\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
where $k = 60$, and $r_m(d)$ is document $d$'s rank in model $m$.

### 3. Bi-Encoder vs. Cross-Encoder
* **Bi-Encoder (`all-MiniLM-L6-v2`)**: Encodes queries and documents separately, enabling precomputed vector indices and fast $O(\log N)$ nearest neighbor search.
* **Cross-Encoder (`ms-marco-TinyBERT-L2-v2`)**: Jointly feeds query and document into the transformer self-attention layers to capture full cross-attention token interactions for higher precision re-ranking.

---

## 🛠 Tech Stack

* **Language:** Python 3.11+
* **NLP & Text Processing:** NLTK (PorterStemmer)
* **Embeddings & Neural Search:** `sentence-transformers`, HuggingFace Hub, PyTorch
* **Computer Vision:** `Pillow`, CLIP (`clip-ViT-B-32`)
* **LLM & Inference:** OpenAI SDK, OpenRouter API
* **Mathematics & Serialization:** NumPy, Pickle, JSON
* **Package Management:** `uv`

---

## 📄 License
This project is licensed under the MIT License.