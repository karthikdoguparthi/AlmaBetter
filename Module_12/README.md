# TourismGPT

A fine-tuned travel chatbot built on **Phi-3 Mini** with **Retrieval-Augmented Generation (RAG)** using a Wikivoyage knowledge base. Built as a college capstone project.

---

## Architecture

```
User Query
    │
    ▼
FAISS Retrieval ──► Top-3 Wikivoyage Chunks
    │                        │
    ▼                        ▼
Fine-tuned Phi-3 Mini (QLoRA) + RAG Context
    │
    ▼
Gradio Chat UI
```

**Stack:**
- **Base model:** Phi-3 Mini 4K Instruct (3.85B parameters)
- **Fine-tuning:** QLoRA via Unsloth (rank=16, 4-bit quantization, T4 GPU)
- **Knowledge base:** English Wikivoyage dump (~5,000 articles, 18,785 chunks)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector store:** FAISS (IndexFlatIP, cosine similarity)
- **UI:** Gradio

---

## Datasets

### Fine-tuning Dataset

2,220 instruction-output pairs across 6 intent categories, built from 3 sources:

| Intent | Source | Pairs |
|---|---|---|
| `itinerary_planning` | NLPC-UOM Travel-Dataset-5000 | ~200 |
| `destination_comparison` | Kaggle Hotel Reviews | ~200 |
| `budget_estimation` | NLPC-UOM Travel-Dataset-5000 | ~200 |
| `hotel_booking_help` | Kaggle Hotel Reviews | ~200 |
| `cancellation_refund_support` | Bitext Customer Support | ~200 |
| `local_customs_safety` | NLPC-UOM Travel-Dataset-5000 | ~200 |

**Sources:**

| Dataset | Link | Used for |
|---|---|---|
| Kaggle Hotel Reviews (Joe Beach) | [kaggle.com/datasets/joebeachcapital/hotel-reviews](https://www.kaggle.com/datasets/joebeachcapital/hotel-reviews) | `hotel_booking_help`, `destination_comparison` — hotel ratings, guest reviews, city-level comparisons |
| Bitext Customer Support LLM Dataset | [huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | `cancellation_refund_support` — cancel/refund/change booking dialogues, rewritten with travel terminology |
| NLPC-UOM Travel Dataset 5000 | [huggingface.co/datasets/NLPC-UOM/Travel-Dataset-5000](https://huggingface.co/datasets/NLPC-UOM/Travel-Dataset-5000) | `itinerary_planning`, `budget_estimation`, `local_customs_safety` — travel Q&A covering planning, costs, and cultural tips |

### RAG Knowledge Base

| Source | Link | Used for |
|---|---|---|
| English Wikivoyage XML Dump | [dumps.wikimedia.org/enwikivoyage/latest/enwikivoyage-latest-pages-articles.xml.bz2](https://dumps.wikimedia.org/enwikivoyage/latest/enwikivoyage-latest-pages-articles.xml.bz2) | Destination articles parsed into 18,785 chunks, embedded and stored in FAISS for retrieval at inference time. Provides factual, up-to-date knowledge about destinations, attractions, transport, and local customs. License: CC BY-SA 3.0 |

> **Note:** The Wikivoyage dump is not included in this repository due to file size (123 MB). Download it from the link above and place it in the project root before running `rag_pipeline.ipynb`.

---

## Repository Structure

```
cap12/
├── archive/                        # TripAdvisor source CSVs
│   ├── reviews.csv
│   └── offerings.csv
├── pipeline.py                     # Dataset pipeline (3 sources → JSONL)
├── generate_ai_pairs.py            # Additional AI pair generation
├── tourism_finetune.jsonl          # Final training dataset (2,220 pairs)
├── finetune_phi3.ipynb             # Fine-tuning notebook (Colab)
├── rag_pipeline.ipynb              # RAG pipeline notebook (Colab)
├── tourismgpt_ui.ipynb             # Gradio chat UI notebook (Colab)
└── enwikivoyage-latest-pages-articles.xml.bz2  # Wikivoyage dump
```

---

## Setup & Usage

All notebooks are designed to run on **Google Colab (T4 GPU, free tier)**.

### Step 1 — Build the dataset

```bash
pip install datasets huggingface_hub pandas
python pipeline.py          # generates tourism_finetune.jsonl
```

### Step 2 — Fine-tune Phi-3 Mini

1. Upload `tourism_finetune.jsonl` to Google Drive
2. Open `finetune_phi3.ipynb` in Colab
3. Run all cells — training takes ~18 min on T4
4. Adapter saved automatically to `MyDrive/tourism_gpt_adapter/`

**Training config:**
- LoRA rank: 16, alpha: 16
- Batch size: 8 (2 × 4 gradient accumulation)
- Steps: 300 (~2 epochs)
- Learning rate: 2e-4 with cosine scheduler
- Peak VRAM: 5.6 GB / 14.6 GB

### Step 3 — Build the RAG index

1. Upload `enwikivoyage-latest-pages-articles.xml.bz2` to Google Drive
2. Open `rag_pipeline.ipynb` in Colab
3. Run all cells — parses 5,000 articles, builds FAISS index (~5 min)
4. Index saved automatically to `MyDrive/wikivoyage_index/`

### Step 4 — Launch the chat UI

1. Open `tourismgpt_ui.ipynb` in Colab
2. Run all cells
3. A public Gradio link appears — share it for demo

---

## Results

Fine-tuning loss curve:

| Step | Loss |
|---|---|
| 25 | 1.729 |
| 50 | 1.067 |
| 75 | 0.793 |
| 300 | converged |

Sample responses (no RAG):

**Q:** What cultural customs should I know before visiting Japan?  
**A:** • Remove shoes before entering homes and many restaurants • Do not tip — it can be considered rude • Bow slightly when greeting • Avoid eating or drinking while walking • Use two hands when giving or receiving business cards

**Q:** What is the estimated budget for a week in Paris?  
**A:** Budget travellers can expect to spend $50–80/day covering accommodation, meals, and transport. Mid-range budgets of $100–150/day allow more comfort. Always set aside 10–15% for unexpected expenses.

---

## Limitations

- Hotel booking and refund questions rely on fine-tuned knowledge only (Wikivoyage does not cover booking platforms or refund procedures)
- RAG retrieval quality depends on query-article semantic overlap; abstract questions retrieve weakly
- Model responses are generated, not verified — always cross-check travel information

---

## Requirements

```
unsloth[colab-new]
trl<0.9.0
peft
accelerate
bitsandbytes
xformers<0.0.27
datasets
huggingface_hub
sentence-transformers
faiss-gpu
gradio
pandas
```

---

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) — fast QLoRA fine-tuning
- [Microsoft Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) — base model
- [Unsloth](https://github.com/unslothai/unsloth) — fast QLoRA fine-tuning
- [Microsoft Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) — base model
- [Wikivoyage](https://en.wikivoyage.org) — RAG knowledge base (CC BY-SA 3.0)
- [Kaggle Hotel Reviews](https://www.kaggle.com/datasets/joebeachcapital/hotel-reviews) — hotel booking & destination comparison pairs
- [Bitext Customer Support LLM Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) — cancellation & refund pairs
- [NLPC-UOM Travel Dataset 5000](https://huggingface.co/datasets/NLPC-UOM/Travel-Dataset-5000) — itinerary, budget & customs pairs
