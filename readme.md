# Medical RAG System

A local Retrieval-Augmented Generation (RAG) system built on biomedical data using:

- PubMed XML articles
- ChromaDB (vector database)
- PubMedBERT embeddings
- Ollama (Mistral LLM)

---

## Project Structure

medical-rag/
├── app/
│   ├── build_index.py
│   ├── main.py
│   ├── rag_system.py
│   ├── ingest_pubmed_xml.py
│   └── requirements.txt
├── data/
│   ├── raw/
│   │   └── articles.xml
│   └── chroma/
├── .env
└── README.md

---

## 1. Prerequisites

- Python 3.10+
- pip
- git
- Ollama

---

## 2. Setup Virtual Environment

python -m venv revenv
source revenv/bin/activate

---

## 3. Install Dependencies

pip install -r app/requirements.txt

pip install torch torchvision torchaudio

---

## 4. Install Ollama

curl -fsSL https://ollama.com/install.sh | sh

ollama --version

---

## 5. Start Ollama

ollama serve

---

## 6. Download Model

ollama pull mistral

---

## 7. Prepare Data

Put your XML inside:

data/raw/articles.xml

IMPORTANT:
Use folder path:

--xml_path data/raw

---

## 8. Build the Index

rm -rf data/chroma
mkdir -p data/chroma

python -m app.build_index --xml_path data/raw --persist_dir data/chroma

---

## 9. Run the RAG App

python -m app.main --persist_dir data/chroma

---

## Full Run Workflow

source revenv/bin/activate
ollama serve

# new terminal
source revenv/bin/activate
ollama pull mistral

rm -rf data/chroma
mkdir -p data/chroma

python -m app.build_index --xml_path data/raw --persist_dir data/chroma
python -m app.main --persist_dir data/chroma

---

## Troubleshooting

NotADirectoryError → use folder, not file  
Embedding error → remove show_progress_bar  
Chroma error → remove HNSW metadata  
Ollama missing → ollama pull mistral  

---

## Quick Start

python -m venv revenv
source revenv/bin/activate

pip install -r app/requirements.txt
pip install torch torchvision torchaudio

curl -fsSL https://ollama.com/install.sh | sh
ollama serve

# new terminal
source revenv/bin/activate

ollama pull mistral

rm -rf data/chroma
mkdir -p data/chroma

python -m app.build_index --xml_path data/raw --persist_dir data/chroma
python -m app.main --persist_dir data/chroma
