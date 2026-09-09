# Local Manual RAG Assistant

A local retrieval-augmented generation (RAG) assistant for searching and answering questions from the American Airlines Customer Service Manual. It retrieves the most relevant manual chunks with embeddings, then asks a local Ollama chat model to answer using that context only.

## Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally

Pull the models used by the project:

```bash
ollama pull embeddinggemma
ollama pull qwen3.5
```

## Setup

Install the Python dependencies:

```bash
uv sync
```

The embedder uses NumPy. If it is not already available in your environment, add it to the project first:

```bash
uv add numpy
```

## Manual data

Place the processed manual chunks at `data/chunks.json`. The file must contain a JSON array, where each item has at least a `text` field.

```json
[
  {
    "id": 1,
    "chapter": 3,
    "section": "3.6.3",
    "title": "Example policy title",
    "text": "The searchable text for this manual section."
  }
]
```

## Run

Start the interactive assistant:

```bash
uv run python llm.py
```

Enter a question at the `>>>` prompt. The assistant retrieves the five most similar chunks and supplies them to `qwen3.5` as context. Enter `N` when prompted to end the session.

## How it works

1. `embedder.py` embeds every `text` chunk using Ollama's `embeddinggemma` model.
2. It embeds your question and ranks chunks by cosine similarity.
3. `llm.py` sends the five best matches, together with the question, to the local `qwen3.5` model.
4. The system prompt instructs the model to answer only from retrieved manual content and to flag insufficient context.

## Current limitations

- Chunks are embedded again for every question, so a large manual will be slow and repeatedly call Ollama.
- Retrieval currently returns a fixed five chunks and does not persist embeddings.
- The expected Ollama host is `http://localhost:11434`.
